import os
import discord
from discord import app_commands
from discord.ext import commands, tasks
from blockfrost import BlockFrostApi, ApiError, ApiUrls
from dotenv import load_dotenv
import asyncio
from datetime import datetime, timezone
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from database import Database, TokenTracker as DbTokenTracker

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
db = Database(os.getenv('DATABASE_URL'))

# Initialize Discord bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Store active tracking configurations
active_trackers = {}

class TokenTracker:
    def __init__(self, policy_id, image_url, threshold, channel_id):
        self.policy_id = policy_id
        self.image_url = image_url
        self.threshold = threshold
        self.channel_id = channel_id
        self.last_block = None
        self.token_info = None
        self.total_volume_24h = 0
        self.transactions_24h = 0
        self.track_transfers = True
        logger.info(f"Created new TokenTracker for policy_id: {policy_id}")
        
        # Save to database
        try:
            db.save_token_tracker({
                'policy_id': policy_id,
                'image_url': image_url,
                'threshold': threshold,
                'channel_id': channel_id,
                'last_block': None,
                'track_transfers': True
            })
        except Exception as e:
            logger.error(f"Failed to save token tracker to database: {str(e)}", exc_info=True)

async def get_token_info(api: BlockFrostApi, policy_id: str):
    try:
        logger.info(f"Fetching token info for policy_id: {policy_id}")
        
        # Get assets for the policy ID (this returns a list directly, no need to await)
        assets = api.assets_policy(policy_id=policy_id)
        if isinstance(assets, Exception):
            raise assets
        
        if not assets:
            logger.warning(f"No assets found for policy_id: {policy_id}")
            return None
            
        # Get the first asset's details
        asset = assets[0]
        asset_id = f"{policy_id}{asset.asset_name.hex() if asset.asset_name else ''}"
        
        # Get asset details
        asset_details = api.asset(asset_id)
        if isinstance(asset_details, Exception):
            raise asset_details
        
        if not asset_details:
            logger.warning(f"No details found for asset: {asset_id}")
            return None
            
        token_info = {
            'asset_id': asset_id,
            'name': asset_details.onchain_metadata.get('name', 'Unknown Token') if asset_details.onchain_metadata else 'Unknown Token',
            'description': asset_details.onchain_metadata.get('description', '') if asset_details.onchain_metadata else '',
            'image': asset_details.onchain_metadata.get('image', '') if asset_details.onchain_metadata else ''
        }
        
        logger.info(f"Successfully fetched token info: {token_info}")
        return token_info
        
    except Exception as e:
        logger.error(f"Unexpected error in get_token_info: {str(e)}", exc_info=True)
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
        embed.set_footer(text=f"Transaction detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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

@bot.tree.command(name="start", description="Start tracking token purchases and transfers")
async def start(interaction: discord.Interaction, policy_id: str, image_url: str, threshold: float, track_transfers: bool = True):
    try:
        logger.info(f"Starting token tracking for policy_id: {policy_id}")
        embed = discord.Embed(
            title="🚀 Token Tracking Initialized",
            description="Successfully started monitoring token activity on the Cardano blockchain.",
            color=discord.Color.brand_green()
        )
        
        tracking_types = []
        if threshold > 0:
            tracking_types.append("Token Purchases")
        if track_transfers:
            tracking_types.append("Wallet Transfers")
        
        embed.add_field(
            name="🎯 Tracking Types",
            value=f"```{', '.join(tracking_types)}```",
            inline=False
        )
        
        embed.add_field(
            name="📋 Token Policy ID",
            value=f"`{policy_id}`",
            inline=False
        )
        
        if threshold > 0:
            embed.add_field(
                name="💎 Purchase Threshold",
                value=f"```{threshold:,.2f} ADA```",
                inline=False
            )
        
        embed.set_footer(text="Tracking started at")
        embed.timestamp = discord.utils.utcnow()
        embed.set_thumbnail(url=image_url)
        
        # Store tracking configuration
        tracker = TokenTracker(
            policy_id=policy_id,
            image_url=image_url,
            threshold=threshold,
            channel_id=interaction.channel_id
        )
        tracker.track_transfers = track_transfers
        active_trackers[policy_id] = tracker
        
        logger.info(f"Token tracking started successfully for policy_id: {policy_id}")
        await interaction.response.send_message(embed=embed)

    except Exception as e:
        logger.error(f"Error starting tracker: {str(e)}", exc_info=True)
        error_embed = discord.Embed(
            title="❌ Error Starting Tracker",
            description=f"```{str(e)}```",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=error_embed, ephemeral=True)

@bot.tree.command(name="token_info", description="Get detailed information about the tracked token")
async def token_info(interaction: discord.Interaction, policy_id: str):
    try:
        api = BlockFrostApi(
            project_id=os.getenv('BLOCKFROST_API_KEY'),
            base_url=ApiUrls.mainnet.value
        )
        
        token_info = await get_token_info(api, policy_id)
        if not token_info:
            raise ValueError("Token not found")
        
        embed = discord.Embed(
            title="📊 Token Information",
            description=f"Detailed information for token under policy ID: `{policy_id}`",
            color=discord.Color.blue()
        )
        
        # Token Details
        embed.add_field(
            name="🪙 Token Name",
            value=f"```{token_info.asset_name.decode() if token_info.asset_name else 'N/A'}```",
            inline=True
        )
        
        embed.add_field(
            name="📈 Initial Mint Tx",
            value=f"[View Transaction](https://cardanoscan.io/transaction/{token_info.initial_mint_tx_hash})",
            inline=True
        )
        
        # Supply Information
        embed.add_field(
            name="💎 Total Supply",
            value=f"```{int(token_info.quantity):,}```",
            inline=False
        )
        
        if token_info.metadata:
            if token_info.metadata.get('description'):
                embed.add_field(
                    name="📝 Description",
                    value=f"```{token_info.metadata['description']}```",
                    inline=False
                )
            
            if token_info.metadata.get('ticker'):
                embed.add_field(
                    name="🏷️ Ticker",
                    value=f"```{token_info.metadata['ticker']}```",
                    inline=True
                )
        
        # Add timestamp
        embed.timestamp = discord.utils.utcnow()
        embed.set_footer(text="Data retrieved at")
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error fetching token info: {str(e)}", exc_info=True)
        error_embed = discord.Embed(
            title="❌ Error Fetching Token Info",
            description=f"```{str(e)}```",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=error_embed, ephemeral=True)

@bot.tree.command(name="volume", description="Get 24-hour trading volume for the tracked token")
async def volume(interaction: discord.Interaction, policy_id: str):
    try:
        if policy_id not in active_trackers:
            raise ValueError("This token is not being tracked")
            
        tracker = active_trackers[policy_id]
        
        embed = discord.Embed(
            title="📊 24-Hour Trading Statistics",
            description=f"Trading activity for the last 24 hours",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="💰 Total Volume",
            value=f"```{tracker.total_volume_24h:,.2f} ADA```",
            inline=True
        )
        
        embed.add_field(
            name="🔄 Total Transactions",
            value=f"```{tracker.transactions_24h:,}```",
            inline=True
        )
        
        if tracker.transactions_24h > 0:
            avg_transaction = tracker.total_volume_24h / tracker.transactions_24h
            embed.add_field(
                name="📈 Average Transaction",
                value=f"```{avg_transaction:,.2f} ADA```",
                inline=True
            )
        
        embed.timestamp = discord.utils.utcnow()
        embed.set_footer(text="Statistics as of")
        embed.set_thumbnail(url=tracker.image_url)
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error fetching volume statistics: {str(e)}", exc_info=True)
        error_embed = discord.Embed(
            title="❌ Error Fetching Volume Statistics",
            description=f"```{str(e)}```",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=error_embed, ephemeral=True)

@tasks.loop(seconds=60)
async def check_transactions():
    try:
        logger.debug("Starting transaction check cycle")
        api = BlockFrostApi(
            project_id=os.getenv('BLOCKFROST_API_KEY'),
            base_url=ApiUrls.mainnet.value
        )
        
        for policy_id, tracker in active_trackers.items():
            try:
                if not tracker.last_block:
                    # Get latest block height
                    latest_block = api.block_latest()
                    if isinstance(latest_block, Exception):
                        raise latest_block
                    tracker.last_block = latest_block.height
                    logger.info(f"Set initial block height for {policy_id}: {tracker.last_block}")
                    
                    # Update database with initial block height
                    try:
                        db.update_last_block(policy_id, tracker.channel_id, tracker.last_block)
                    except Exception as e:
                        logger.error(f"Failed to update last block in database: {str(e)}", exc_info=True)
                    continue

                # Get transactions since last check
                transactions = api.address_transactions(policy_id, from_block=tracker.last_block)
                if isinstance(transactions, Exception):
                    raise transactions
                logger.info(f"Found {len(transactions)} new transactions for {policy_id}")
                
                for tx in transactions:
                    try:
                        tx_details = api.transaction(tx.tx_hash)
                        if isinstance(tx_details, Exception):
                            raise tx_details
                            
                        tx_type, ada_amount, token_amount, analysis_details = await analyze_transaction_improved(tx_details, policy_id)
                        
                        if tx_type == 'dex_trade' and ada_amount >= tracker.threshold:
                            logger.info(f"Found DEX trade transaction: {tx.tx_hash}")
                            embed = await create_trade_embed(
                                tx_details, policy_id, ada_amount, token_amount, tracker, analysis_details
                            )
                            if embed:
                                channel = bot.get_channel(tracker.channel_id)
                                if channel:
                                    await channel.send(embed=embed)
                                    
                        elif tx_type == 'wallet_transfer' and tracker.track_transfers:
                            logger.info(f"Found wallet transfer transaction: {tx.tx_hash}")
                            transfer_embed = await create_transfer_embed(
                                tx_details, policy_id, token_amount, tracker
                            )
                            if transfer_embed:
                                channel = bot.get_channel(tracker.channel_id)
                                if channel:
                                    await channel.send(embed=transfer_embed)
                    
                    except Exception as tx_e:
                        logger.error(f"Error processing transaction {tx.tx_hash}: {str(tx_e)}", exc_info=True)
                        continue

                # Update last block height
                latest_block = api.block_latest()
                if isinstance(latest_block, Exception):
                    raise latest_block
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
        logger.error(f"Error in check_transactions task: {str(e)}", exc_info=True)

@bot.event
async def on_ready():
    logger.info(f"Bot is ready: {bot.user}")
    
    # Load trackers from database
    try:
        saved_trackers = db.get_all_token_trackers()
        for tracker_data in saved_trackers:
            tracker = TokenTracker(
                policy_id=tracker_data['policy_id'],
                image_url=tracker_data.get('image_url'),
                threshold=tracker_data['threshold'],
                channel_id=tracker_data['channel_id']
            )
            tracker.last_block = tracker_data.get('last_block')
            tracker.track_transfers = tracker_data.get('track_transfers', True)
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
