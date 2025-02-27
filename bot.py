import os
import discord
from discord import app_commands
from discord.ext import commands, tasks
from blockfrost import BlockFrostApi, ApiError, ApiUrls
from dotenv import load_dotenv
import asyncio
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
import database as db

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler for all logs
    file_handler = RotatingFileHandler(
        'logs/bot.log',
        maxBytes=10000000,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_format)

    # File handler for errors only
    error_handler = RotatingFileHandler(
        'logs/error.log',
        maxBytes=10000000,  # 10MB
        backupCount=5
    )
    error_handler.setFormatter(log_format)
    error_handler.setLevel(logging.ERROR)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    # Get the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize logging
logger = setup_logging()

# Load environment variables
load_dotenv()

# Initialize database connection
db = db.Database(os.getenv('DATABASE_URL'))

# Initialize Discord bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Store active tracking configurations
active_trackers = {}

class TokenTracker:
    def __init__(self, policy_id, token_name, image_url, threshold, channel_id, last_block=None, track_transfers=True, trade_notifications=0, transfer_notifications=0):
        self.policy_id = policy_id
        self.token_name = token_name
        self.image_url = image_url
        self.threshold = threshold
        self.channel_id = channel_id
        self.last_block = last_block
        self.track_transfers = track_transfers
        self.trade_notifications = trade_notifications
        self.transfer_notifications = transfer_notifications
        
        # Save to database
        try:
            db.save_token_tracker({
                'policy_id': policy_id,
                'token_name': token_name,
                'image_url': image_url,
                'threshold': threshold,
                'channel_id': channel_id,
                'last_block': last_block,
                'track_transfers': track_transfers,
                'trade_notifications': trade_notifications,
                'transfer_notifications': transfer_notifications
            })
        except Exception as e:
            logger.error(f"Failed to save token tracker to database: {str(e)}", exc_info=True)
        
        logger.info(f"Created new TokenTracker for {token_name} (policy_id: {policy_id})")
        
    def increment_trade_notifications(self):
        self.trade_notifications += 1
        db.update_notification_counts(self.policy_id, self.channel_id, self.trade_notifications, self.transfer_notifications)

    def increment_transfer_notifications(self):
        self.transfer_notifications += 1
        db.update_notification_counts(self.policy_id, self.channel_id, self.trade_notifications, self.transfer_notifications)

async def get_token_info(api: BlockFrostApi, policy_id: str):
    try:
        # Get all assets under this policy
        assets = await api.assets_policy(policy_id)
        if isinstance(assets, Exception):
            raise assets
        return assets[0] if assets else None
    except Exception as e:
        logger.error(f"Error getting token info: {str(e)}", exc_info=True)
        return None

async def get_transaction_details(api: BlockFrostApi, tx_hash: str):
    try:
        logger.info(f"Fetching transaction details for tx_hash: {tx_hash}")
        # Get detailed transaction information
        tx = api.transaction(tx_hash)
        if isinstance(tx, Exception):
            raise tx
        logger.debug(f"Transaction details retrieved: {tx_hash}")
        
        # Get transaction UTXOs
        utxos = api.transaction_utxos(tx_hash)
        if isinstance(utxos, Exception):
            raise utxos
        logger.debug(f"Transaction UTXOs retrieved: {tx_hash}")
        
        # Get transaction metadata if available
        metadata = api.transaction_metadata(tx_hash)
        if isinstance(metadata, Exception):
            raise metadata
        logger.debug(f"Transaction metadata retrieved: {tx_hash}")
        
        return tx, utxos, metadata
    except ApiError as e:
        logger.error(f"Blockfrost API error while fetching transaction details: {str(e)}")
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error in get_transaction_details: {str(e)}", exc_info=True)
        return None, None, None

async def analyze_transaction_improved(tx_details, policy_id):
    """
    Enhanced transaction analysis that detects DEX trades by analyzing transaction patterns
    Returns: (type, ada_amount, token_amount, details)
    type can be 'dex_trade', 'wallet_transfer', or None
    """
    try:
        # Initialize transaction analysis
        input_addresses = set()
        output_addresses = set()
        input_tokens = 0
        output_tokens = 0
        input_ada = 0
        output_ada = 0
        
        # Analyze inputs
        for inp in tx_details.inputs:
            input_addresses.add(inp.address)
            for amount in inp.amount:
                if amount.unit.startswith(policy_id):
                    input_tokens += int(amount.quantity)
                elif amount.unit == "lovelace":
                    input_ada += int(amount.quantity) / 1_000_000

        # Analyze outputs
        for out in tx_details.outputs:
            output_addresses.add(out.address)
            for amount in out.amount:
                if amount.unit.startswith(policy_id):
                    output_tokens += int(amount.quantity)
                elif amount.unit == "lovelace":
                    output_ada += int(amount.quantity) / 1_000_000

        # Calculate net movements
        ada_movement = abs(output_ada - input_ada)
        token_movement = abs(output_tokens - input_tokens)
        
        # Transaction pattern analysis
        details = {
            "addresses": {
                "input": list(input_addresses),
                "output": list(output_addresses)
            },
            "movements": {
                "ada": ada_movement,
                "tokens": token_movement
            }
        }

        # DEX Trade Pattern Detection:
        # 1. Significant ADA movement (> 1 ADA)
        # 2. Token amount changes
        # 3. Multiple addresses involved (typically DEX contracts)
        # 4. Complex transaction structure
        is_likely_dex = (
            ada_movement > 1 and  # Significant ADA movement
            len(input_addresses) + len(output_addresses) > 3 and  # Multiple addresses involved
            (
                (input_tokens == 0 and output_tokens > 0) or  # Buying tokens
                (input_tokens > 0 and output_tokens == 0)     # Selling tokens
            )
        )
        
        if is_likely_dex:
            trade_type = "buy" if output_tokens > input_tokens else "sell"
            details["trade_type"] = trade_type
            return 'dex_trade', ada_movement, token_movement, details
        
        # Wallet Transfer Pattern:
        # 1. Minimal ADA movement (just fees)
        # 2. Tokens moving between addresses
        # 3. Simple transaction structure
        elif input_tokens > 0 and output_tokens > 0:  # Tokens moving
            if input_addresses != output_addresses:  # Different addresses
                if ada_movement <= 1:  # Only fee-level ADA movement
                    return 'wallet_transfer', ada_movement, token_movement, details

        return None, 0, 0, details

    except Exception as e:
        logger.error(f"Error in improved transaction analysis: {str(e)}", exc_info=True)
        return None, 0, 0, {"error": str(e)}

def create_pool_pm_link(address):
    """Creates a pool.pm link for a Cardano address"""
    return f"[{address[:8]}...{address[-4:]}](https://pool.pm/addresses/{address})"

async def create_trade_embed(tx_details, policy_id, ada_amount, token_amount, tracker, analysis_details):
    """Creates a detailed embed for DEX trades with transaction information"""
    try:
        trade_type = analysis_details.get("trade_type", "unknown")
        title_emoji = "💰" if trade_type == "buy" else "💱"
        action_word = "Purchase" if trade_type == "buy" else "Sale"
        
        # Format addresses for display
        input_addresses = analysis_details['addresses']['input']
        output_addresses = analysis_details['addresses']['output']
        
        # Find the main wallet address
        main_wallet = None
        if trade_type == "buy":
            for addr in output_addresses:
                for amount in tx_details.outputs:
                    if amount.address == addr:
                        for token in amount.amount:
                            if token.unit.startswith(policy_id) and int(token.quantity) > 0:
                                main_wallet = addr
                                break
        else:
            for addr in input_addresses:
                for amount in tx_details.inputs:
                    if amount.address == addr:
                        for token in amount.amount:
                            if token.unit.startswith(policy_id) and int(token.quantity) > 0:
                                main_wallet = addr
                                break
        
        embed = discord.Embed(
            title=f"{title_emoji} Token {action_word} Detected",
            description=(
                f"Transaction Hash: [`{tx_details.hash[:8]}...{tx_details.hash[-8:]}`](https://pool.pm/tx/{tx_details.hash})\n"
                f"Main Wallet: {create_pool_pm_link(main_wallet) if main_wallet else 'Unknown'}"
            ),
            color=discord.Color.green() if trade_type == "buy" else discord.Color.blue()
        )

        # Left side: Transaction Overview
        overview = (
            "```\n"
            f"Type     : DEX {action_word}\n"
            f"Block    : {tx_details.block}\n"
            f"Status   : Confirmed\n"
            f"Addresses: {len(input_addresses) + len(output_addresses)}\n"
            "```"
        )
        embed.add_field(
            name="📝 Overview",
            value=overview,
            inline=True
        )

        # Right side: Trade Information
        if trade_type == "buy":
            trade_info = (
                "```\n"
                f"ADA Spent  : {ada_amount:,.2f}\n"
                f"Tokens Recv: {token_amount:,}\n"
                f"Price/Token: {(ada_amount/token_amount):.6f}\n"
                "```"
            )
        else:
            trade_info = (
                "```\n"
                f"Tokens Sold: {token_amount:,}\n"
                f"ADA Recv   : {ada_amount:,.2f}\n"
                f"Price/Token: {(ada_amount/token_amount):.6f}\n"
                "```"
            )
        
        embed.add_field(
            name="💰 Trade Details",
            value=trade_info,
            inline=True
        )

        # Add token info if available
        if tracker.token_info:
            token_name = tracker.token_info.get('name', 'Unknown Token')
            embed.set_author(name=token_name)
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)

        # Format addresses in two columns
        address_layout = []
        
        # Input Addresses Column
        in_addrs = []
        if input_addresses:
            in_addrs.append("📥 Input Addresses:")
            for addr in input_addresses[:3]:
                in_addrs.append(create_pool_pm_link(addr))
            if len(input_addresses) > 3:
                in_addrs.append(f"...and {len(input_addresses) - 3} more")
        
        # Output Addresses Column
        out_addrs = []
        if output_addresses:
            out_addrs.append("📤 Output Addresses:")
            for addr in output_addresses[:3]:
                out_addrs.append(create_pool_pm_link(addr))
            if len(output_addresses) > 3:
                out_addrs.append(f"...and {len(output_addresses) - 3} more")

        # Combine columns with padding
        max_lines = max(len(in_addrs), len(out_addrs))
        for i in range(max_lines):
            left = in_addrs[i] if i < len(in_addrs) else ""
            right = out_addrs[i] if i < len(out_addrs) else ""
            # Add padding to align columns
            address_layout.append(f"{left:<40} {right}")

        if address_layout:
            embed.add_field(
                name="🔍 Addresses",
                value="\n".join(address_layout),
                inline=False
            )

        # Add policy ID in code block
        embed.add_field(
            name="🔑 Policy ID",
            value=f"```{policy_id}```",
            inline=False
        )

        # Add footer with timestamp
        embed.set_footer(text=f"Transaction detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return embed

    except Exception as e:
        logger.error(f"Error creating trade embed: {str(e)}", exc_info=True)
        return None

async def create_transfer_embed(tx_details, policy_id, token_amount, tracker):
    """Creates an embed for token transfer notifications"""
    try:
        embed = discord.Embed(
            title="↔️ Token Transfer Detected",
            description="Tokens have been transferred between wallets.",
            color=discord.Color.blue()
        )

        # From/To Addresses
        from_address = tx_details.inputs[0].address
        to_address = tx_details.outputs[0].address
        
        transfer_details = (
            f"**From:** ```{from_address[:20]}...{from_address[-8:]}```\n"
            f"**To:** ```{to_address[:20]}...{to_address[-8:]}```\n"
            f"**Amount:** ```{token_amount:,} Tokens```"
        )
        embed.add_field(
            name="🔄 Transfer Details",
            value=transfer_details,
            inline=False
        )

        # Add wallet links
        wallet_links = (
            f"[View Sender Wallet](https://cardanoscan.io/address/{from_address})\n"
            f"[View Receiver Wallet](https://cardanoscan.io/address/{to_address})"
        )
        embed.add_field(
            name="👤 Wallet Profiles",
            value=wallet_links,
            inline=False
        )

        # Transaction link
        embed.add_field(
            name="🔍 Transaction Details",
            value=f"[View on CardanoScan](https://cardanoscan.io/transaction/{tx_details.hash})",
            inline=False
        )

        # Set metadata
        embed.set_thumbnail(url=tracker.image_url)
        embed.timestamp = discord.utils.utcnow()
        embed.set_footer(
            text=f"Transfer detected at • Block #{tx_details.block_height}",
            icon_url="https://cardanoscan.io/images/favicon.ico"
        )

        return embed

    except Exception as e:
        logger.error(f"Error creating transfer embed: {str(e)}", exc_info=True)
        return None

def shorten_address(address):
    """Shortens a Cardano address for display"""
    if not address:
        return "Unknown"
    return address[:8] + "..." + address[-4:] if len(address) > 12 else address

class TokenControls(discord.ui.View):
    def __init__(self, policy_id: str):
        super().__init__(timeout=None)  # Buttons don't timeout
        self.policy_id = policy_id

    @discord.ui.button(label="🛑 Stop Tracking", style=discord.ButtonStyle.danger, custom_id="stop_tracking")
    async def stop_tracking(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Stop tracking the token"""
        try:
            if self.policy_id in active_trackers:
                # Remove from database
                try:
                    db.delete_token_tracker(self.policy_id, interaction.channel_id)
                except Exception as e:
                    logger.error(f"Failed to delete token tracker from database: {str(e)}", exc_info=True)
                
                # Remove from memory
                del active_trackers[self.policy_id]
                
                embed = discord.Embed(
                    title="🛑 Tracking Stopped",
                    description=f"Successfully stopped tracking token with policy ID: `{self.policy_id}`",
                    color=discord.Color.red()
                )
                await interaction.response.edit_message(embed=embed, view=None)
                logger.info(f"Stopped tracking token: {self.policy_id}")
            else:
                await interaction.response.send_message("Not tracking this token anymore.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in stop tracking button: {str(e)}", exc_info=True)
            await interaction.response.send_message("Failed to stop tracking. Please try again.", ephemeral=True)

    @discord.ui.button(label="🔄 Toggle Transfers", style=discord.ButtonStyle.primary, custom_id="toggle_transfers")
    async def toggle_transfers(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Toggle transfer notifications"""
        try:
            if self.policy_id in active_trackers:
                tracker = active_trackers[self.policy_id]
                tracker.track_transfers = not tracker.track_transfers
                
                # Update database
                try:
                    db.save_token_tracker({
                        'policy_id': self.policy_id,
                        'token_name': tracker.token_name,
                        'image_url': tracker.image_url,
                        'threshold': tracker.threshold,
                        'channel_id': interaction.channel_id,
                        'last_block': tracker.last_block,
                        'track_transfers': tracker.track_transfers
                    })
                except Exception as e:
                    logger.error(f"Failed to update token tracker in database: {str(e)}", exc_info=True)
                
                status = "enabled" if tracker.track_transfers else "disabled"
                await interaction.response.send_message(f"Transfer notifications {status} for this token.", ephemeral=True)
            else:
                await interaction.response.send_message("Not tracking this token anymore.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in toggle transfers button: {str(e)}", exc_info=True)
            await interaction.response.send_message("Failed to toggle transfers. Please try again.", ephemeral=True)

class TokenSetupModal(discord.ui.Modal, title="🪙 Token Setup"):
    def __init__(self):
        super().__init__()
        self.policy_id = discord.ui.TextInput(
            label="Policy ID",
            placeholder="Enter the token's policy ID",
            style=discord.TextStyle.short,
            required=True,
            min_length=56,
            max_length=56
        )
        self.token_name = discord.ui.TextInput(
            label="Token Name",
            placeholder="Enter the token's name",
            style=discord.TextStyle.short,
            required=True
        )
        self.image_url = discord.ui.TextInput(
            label="Image URL",
            placeholder="Enter custom image URL (optional)",
            style=discord.TextStyle.short,
            required=False
        )
        self.threshold = discord.ui.TextInput(
            label="Minimum Token Amount",
            placeholder="Min tokens for notifications (default: 1000)",
            style=discord.TextStyle.short,
            required=False,
            default="1000"
        )
        self.track_transfers = discord.ui.TextInput(
            label="Track Transfers",
            placeholder="Type 'yes' or 'no' (default: yes)",
            style=discord.TextStyle.short,
            required=False,
            default="yes"
        )
        
        self.add_item(self.policy_id)
        self.add_item(self.token_name)
        self.add_item(self.image_url)
        self.add_item(self.threshold)
        self.add_item(self.track_transfers)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            # Create loading message
            loading_embed = discord.Embed(
                title="🔍 Initializing Token Tracking",
                description="Please wait while I set up token tracking...",
                color=discord.Color.blue()
            )
            await interaction.response.send_message(embed=loading_embed)
            message = await interaction.original_response()

            policy_id = self.policy_id.value.strip()
            token_name = self.token_name.value.strip()
            image_url = self.image_url.value.strip() if self.image_url.value else None
            
            # Validate and convert threshold
            try:
                threshold = float(self.threshold.value or "1000")
                if threshold <= 0:
                    raise ValueError("Threshold must be positive")
            except ValueError:
                error_embed = discord.Embed(
                    title="❌ Invalid Threshold",
                    description="Please enter a valid positive number for the threshold.",
                    color=discord.Color.red()
                )
                await message.edit(embed=error_embed)
                return

            # Parse track_transfers
            track_transfers = self.track_transfers.value.lower() != "no"

            # Check if already tracking and remove if exists
            if policy_id in active_trackers:
                # Remove from memory
                old_tracker = active_trackers.pop(policy_id)
                # Remove from database
                try:
                    db.delete_token_tracker(policy_id, old_tracker.channel_id)
                except Exception as e:
                    logger.error(f"Failed to delete old token tracker from database: {str(e)}", exc_info=True)
                logger.info(f"Removed existing tracker for {policy_id}")

            # Create new tracker
            tracker = TokenTracker(
                policy_id=policy_id,
                token_name=token_name,
                image_url=image_url,
                threshold=threshold,
                channel_id=interaction.channel_id
            )
            tracker.track_transfers = track_transfers
            
            # Create success embed
            embed = discord.Embed(
                title="✅ Token Tracking Started",
                description=(
                    f"Successfully initialized tracking for:\n"
                    f"**Token:** {token_name}\n"
                    f"**Policy ID:** `{policy_id}`"
                ),
                color=discord.Color.green()
            )
            
            # Add token image if available
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)
            
            # Add configuration fields
            embed.add_field(
                name="⚙️ Configuration",
                value=(
                    f"**Threshold:** `{threshold:,.2f} Tokens`\n"
                    f"**Channel:** {interaction.channel.mention}\n"
                    f"**Transfer Notifications:** {'Enabled' if track_transfers else 'Disabled'}\n"
                    f"**Image URL:** {tracker.image_url or 'None'}"
                ),
                inline=False
            )
            
            # Add monitoring details
            embed.add_field(
                name="📊 Monitoring",
                value=(
                    "• DEX Trades (Buys/Sells)\n"
                    "• Wallet Transfers\n"
                    "• Price per Token\n"
                    "• Transaction Details"
                ),
                inline=True
            )
            
            # Add notification details
            embed.add_field(
                name="🔔 Notifications",
                value=(
                    "• Trade Amount\n"
                    "• Wallet Addresses\n"
                    "• Block Height\n"
                    "• Transaction Hash"
                ),
                inline=True
            )
            
            # Add footer with timestamp
            embed.set_footer(text=f"Started tracking at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Create view with buttons
            view = TokenControls(policy_id)
            
            # Update the loading message
            await message.edit(embed=embed, view=view)
            
            # Save to memory and database
            active_trackers[policy_id] = tracker
            try:
                db.save_token_tracker({
                    'policy_id': policy_id,
                    'token_name': token_name,
                    'image_url': tracker.image_url,
                    'threshold': threshold,
                    'channel_id': interaction.channel_id,
                    'last_block': None,
                    'track_transfers': track_transfers
                })
            except Exception as e:
                logger.error(f"Failed to save token tracker to database: {str(e)}", exc_info=True)
            
            logger.info(f"Started tracking token: {token_name} ({policy_id})")
                
        except Exception as e:
            logger.error(f"Error in token setup: {str(e)}", exc_info=True)
            error_embed = discord.Embed(
                title="❌ Error",
                description="Failed to start tracking. Please check your inputs and try again.",
                color=discord.Color.red()
            )
            await message.edit(embed=error_embed)

@bot.tree.command(name="start", description="Start tracking token purchases and transfers")
async def start(interaction: discord.Interaction):
    """Start tracking a token by opening a setup form"""
    modal = TokenSetupModal()
    await interaction.response.send_modal(modal)

@bot.tree.command(name="help")
async def help_command(interaction: discord.Interaction):
    """Display help information about the bot's commands"""
    embed = discord.Embed(
        title="🐾 PUP Bot Help",
        description="Track Cardano token transactions with real-time notifications!",
        color=discord.Color.blue()
    )

    # Commands section
    embed.add_field(
        name="📝 Commands",
        value=(
            "**`/start`**\n"
            "Start tracking a token's transactions\n\n"
            "**`/status`**\n"
            "View currently tracked tokens and their settings\n\n"
            "**`/help`**\n"
            "Show this help message"
        ),
        inline=False
    )

    # Features section
    embed.add_field(
        name="🔍 Monitoring Features",
        value=(
            "• DEX Trade Detection\n"
            "• Wallet Transfer Tracking\n"
            "• Real-time Notifications\n"
            "• Customizable Thresholds"
        ),
        inline=True
    )

    # Notification Details
    embed.add_field(
        name="🔔 Notifications Include",
        value=(
            "• Trade Amount\n"
            "• Wallet Addresses\n"
            "• Block Height\n"
            "• Transaction Hash"
        ),
        inline=True
    )

    # Footer
    embed.set_footer(text="Need more help? Contact the bot administrator.")
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Show status of tracked tokens in this channel")
async def status(interaction: discord.Interaction):
    """Show status of tracked tokens in this channel"""
    try:
        # Create embed
        embed = discord.Embed(
            title="Token Tracking Status",
            description="Currently tracked tokens in this channel:",
            color=discord.Color.blue()
        )

        for policy_id, tracker in active_trackers.items():
            # Only show trackers for this channel
            if tracker.channel_id != interaction.channel_id:
                continue

            token_name = tracker.token_name
            
            # Add field for each token
            embed.add_field(
                name=token_name,
                value=(
                    f"Policy ID: `{policy_id}`\n"
                    f"Threshold: `{tracker.threshold:,.2f}`\n"
                    f"Transfers: `{'On' if tracker.track_transfers else 'Off'}`\n"
                    f"Trade Alerts: `{tracker.trade_notifications}`\n"
                    f"Transfer Alerts: `{tracker.transfer_notifications}`"
                ),
                inline=False
            )

            # Set thumbnail to the first token's image
            if tracker.image_url and not embed.thumbnail:
                embed.set_thumbnail(url=tracker.image_url)

        # Add footer with timestamp
        embed.set_footer(text=f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Send the embed without any view/buttons
        await interaction.response.send_message(embed=embed)

    except Exception as e:
        logger.error(f"Error in status command: {str(e)}", exc_info=True)
        await interaction.response.send_message("Failed to get status. Please try again.", ephemeral=True)

@tasks.loop(seconds=60)
async def check_transactions():
    try:
        if not active_trackers:
            return

        api = BlockFrostApi(
            project_id=os.getenv('BLOCKFROST_API_KEY'),
            base_url=ApiUrls.mainnet.value
        )
        
        # Get latest block
        latest_block = await api.block_latest()
        if isinstance(latest_block, Exception):
            raise latest_block

        for policy_id, tracker in active_trackers.items():
            try:
                # Get asset info first
                asset_info = await get_token_info(api, policy_id)
                if not asset_info:
                    logger.error(f"Could not find asset info for policy {policy_id}")
                    continue

                # Construct full asset ID (policy_id + hex encoded asset name)
                asset_name_hex = asset_info.asset_name.hex() if hasattr(asset_info, 'asset_name') else ''
                full_asset_id = f"{policy_id}{asset_name_hex}"
                
                # Get transactions since last check
                transactions = await api.asset_transactions(full_asset_id, from_block=tracker.last_block)
                if isinstance(transactions, Exception):
                    raise transactions
                logger.info(f"Found {len(transactions)} new transactions for {policy_id}")

                # Process each transaction
                for tx in transactions:
                    try:
                        # Get full transaction details
                        tx_details = await get_transaction_details(api, tx.tx_hash)
                        if not tx_details:
                            continue

                        # Analyze the transaction
                        tx_type, ada_amount, token_amount, details = await analyze_transaction_improved(tx_details, policy_id)
                        
                        # For trades, check ADA amount
                        if tx_type == 'dex_trade' and ada_amount >= tracker.threshold:
                            # Create and send trade notification
                            embed = await create_trade_embed(tx_details, policy_id, ada_amount, token_amount, tracker, details)
                            channel = bot.get_channel(tracker.channel_id)
                            if channel:
                                await channel.send(embed=embed)
                                tracker.increment_trade_notifications()
                                    
                        # For transfers, check token amount
                        elif tx_type == 'wallet_transfer' and tracker.track_transfers and token_amount >= tracker.threshold:
                            # Create and send transfer notification
                            transfer_embed = await create_transfer_embed(tx_details, policy_id, token_amount, tracker)
                            channel = bot.get_channel(tracker.channel_id)
                            if channel:
                                await channel.send(embed=transfer_embed)
                                tracker.increment_transfer_notifications()
                    
                    except Exception as tx_e:
                        logger.error(f"Error processing transaction {tx.tx_hash}: {str(tx_e)}", exc_info=True)
                        continue

                # Update last checked block
                tracker.last_block = latest_block.height
                logger.debug(f"Updated last block height for {policy_id}: {tracker.last_block}")
                
                # Update database with new block height
                try:
                    db.update_last_block(policy_id, tracker.channel_id, tracker.last_block)
                except Exception as e:
                    logger.error(f"Failed to update last block in database: {str(e)}", exc_info=True)
                
            except Exception as tracker_e:
                logger.error(f"Error processing tracker {policy_id}: {str(tracker_e)}", exc_info=True)
                continue
                
    except Exception as e:
        logger.error(f"Error in check_transactions: {str(e)}", exc_info=True)

@bot.event
async def on_ready():
    logger.info(f"Bot is ready: {bot.user}")
    
    # Load trackers from database
    try:
        saved_trackers = db.get_all_token_trackers()
        for tracker_data in saved_trackers:
            tracker = TokenTracker(
                policy_id=tracker_data['policy_id'],
                token_name=tracker_data['token_name'],
                image_url=tracker_data.get('image_url'),
                threshold=tracker_data['threshold'],
                channel_id=tracker_data['channel_id'],
                last_block=tracker_data.get('last_block'),
                track_transfers=tracker_data.get('track_transfers', True),
                trade_notifications=tracker_data.get('trade_notifications', 0),
                transfer_notifications=tracker_data.get('transfer_notifications', 0)
            )
            active_trackers[tracker.policy_id] = tracker
            logger.info(f"Loaded tracker from database: {tracker_data['policy_id']}")
    except Exception as e:
        logger.error(f"Failed to load trackers from database: {str(e)}", exc_info=True)

    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s)")
        check_transactions.start()
    except Exception as e:
        logger.error(f"Error syncing commands: {str(e)}", exc_info=True)

# Run the bot
if __name__ == "__main__":
    bot.run(os.getenv('DISCORD_TOKEN'))
