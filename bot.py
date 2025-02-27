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
from database import database
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import hmac
import hashlib
import uvicorn
import threading
import time

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Configure logging for both local development and Heroku"""
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handlers for stdout and stderr
    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setFormatter(log_format)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setFormatter(log_format)
    error_handler.setLevel(logging.WARNING)

    # Get the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # Add handlers
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    # Log startup message
    logger.info("Bot logging initialized")
    
    return logger

# Initialize logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Discord bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Initialize Blockfrost API
api = BlockFrostApi(
    project_id=os.getenv('BLOCKFROST_API_KEY'),
    base_url=ApiUrls.mainnet.value
)

# Webhook secret for verification
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
WEBHOOK_TOLERANCE_SECONDS = 600  # 10 minutes, same as Blockfrost SDK default

def verify_webhook_signature(payload: bytes, header: str, current_time: int) -> bool:
    """
    Verify the Blockfrost webhook signature
    
    Args:
        payload: Raw request body bytes
        header: Blockfrost-Signature header value
        current_time: Current Unix timestamp
    """
    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET not set, skipping signature verification")
        return True
        
    try:
        # Parse header
        pairs = dict(pair.split('=') for pair in header.split(','))
        if 't' not in pairs or 'v1' not in pairs:
            logger.error("Missing timestamp or signature in header")
            return False
            
        # Get timestamp and signature
        timestamp = pairs['t']
        signature = pairs['v1']
        
        # Check timestamp
        timestamp_diff = abs(current_time - int(timestamp))
        if timestamp_diff > WEBHOOK_TOLERANCE_SECONDS:
            logger.error(f"Webhook timestamp too old: {timestamp_diff} seconds")
            return False
            
        # Prepare signature payload
        signature_payload = f"{timestamp}.{payload.decode('utf-8')}"
        
        # Compute expected signature
        computed = hmac.new(
            WEBHOOK_SECRET.encode(),
            signature_payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(computed, signature)
        
    except Exception as e:
        logger.error(f"Error verifying webhook signature: {str(e)}", exc_info=True)
        return False

@app.post("/webhook/transaction")
async def transaction_webhook(request: Request):
    """Handle incoming transaction webhooks from Blockfrost"""
    try:
        # Log webhook received
        logger.info("Received webhook request")
        
        # Get the signature
        signature = request.headers.get('Blockfrost-Signature')
        if not signature:
            logger.error("Missing Blockfrost-Signature header")
            raise HTTPException(status_code=400, detail="Missing signature header")
        
        # Get the raw payload
        payload = await request.body()
        logger.debug(f"Raw payload size: {len(payload)} bytes")
        
        # Verify signature
        current_time = int(time.time())
        if not verify_webhook_signature(payload, signature, current_time):
            logger.error("Invalid webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse the payload
        data = await request.json()
        
        # Log webhook info
        tx_count = len(data.get('payload', []))
        logger.info(f"Webhook contains {tx_count} transaction(s)")
        
        # Validate webhook data
        if not isinstance(data, dict) or 'type' not in data or data['type'] != 'transaction':
            logger.error(f"Invalid webhook data format: {data.get('type', 'unknown type')}")
            return {"status": "ignored"}
            
        # Get transactions from payload
        transactions = data.get('payload', [])
        if not transactions:
            logger.info("Webhook contains no transactions")
            return {"status": "no transactions"}
            
        # Process each transaction
        for tx_data in transactions:
            # Get transaction details
            tx = tx_data.get('tx', {})
            tx_hash = tx.get('hash', 'unknown')
            
            # Log transaction processing
            logger.info(f"Processing transaction: {tx_hash}")
            
            inputs = tx_data.get('inputs', [])
            outputs = tx_data.get('outputs', [])
            
            # Skip if missing required data
            if not tx or not inputs or not outputs:
                logger.warning(f"Skipping transaction {tx_hash} - Missing required data")
                continue
                
            # Check if any of our tracked tokens are involved
            trackers = database.get_trackers()
            logger.info(f"Checking {len(trackers)} tracked tokens")
            
            for tracker in trackers:
                # Check inputs and outputs for our policy ID
                is_involved = False
                
                # Check inputs
                for inp in inputs:
                    for amt in inp.get('amount', []):
                        if amt.get('unit', '').startswith(tracker.policy_id):
                            is_involved = True
                            logger.info(f"Found token {tracker.token_name} in transaction inputs")
                            break
                    if is_involved:
                        break
                        
                # Check outputs if not found in inputs
                if not is_involved:
                    for out in outputs:
                        for amt in out.get('amount', []):
                            if amt.get('unit', '').startswith(tracker.policy_id):
                                is_involved = True
                                logger.info(f"Found token {tracker.token_name} in transaction outputs")
                                break
                        if is_involved:
                            break
                
                if is_involved:
                    # Log token involvement
                    logger.info(f"Analyzing transaction for {tracker.token_name} ({tracker.policy_id})")
                    
                    # Analyze the transaction
                    tx_type, ada_amount, token_amount, details = analyze_transaction_improved(tx_data, tracker.policy_id)
                    
                    # Log analysis results
                    logger.info(f"Analysis results: type={tx_type}, ADA={ada_amount:.2f}, Tokens={token_amount:,}")
                    
                    # Send notification
                    await send_transaction_notification(tracker, tx_type, ada_amount, token_amount, details)
                else:
                    logger.debug(f"Token {tracker.token_name} not involved in transaction {tx_hash}")
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def run_webhook_server():
    """Run the FastAPI webhook server"""
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))

@bot.event
async def on_ready():
    """Called when the bot is ready"""
    logger.info(f"Bot is ready: {bot.user}")
    
    # Start the webhook server in a separate thread
    threading.Thread(target=run_webhook_server, daemon=True).start()
    
    # Load trackers from database
    try:
        saved_trackers = database.get_trackers()  # Using the correct function name
        for tracker in saved_trackers:
            active_trackers[tracker.policy_id] = tracker
            logger.info(f"Loaded tracker for policy {tracker.policy_id}")
    except Exception as e:
        logger.error(f"Failed to load trackers from database: {str(e)}", exc_info=True)

    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s)")
    except Exception as e:
        logger.error(f"Error syncing commands: {str(e)}", exc_info=True)

# Store active tracking configurations
active_trackers = {}

class TokenTracker:
    def __init__(self, policy_id: str, channel_id: int, token_name: str = None, 
                 image_url: str = None, threshold: float = 1000.0, 
                 track_transfers: bool = True, last_block: int = None,
                 trade_notifications: int = 0, transfer_notifications: int = 0):
        self.policy_id = policy_id
        self.channel_id = channel_id
        self.token_name = token_name
        self.image_url = image_url
        self.threshold = threshold
        self.track_transfers = track_transfers
        self.last_block = last_block
        self.trade_notifications = trade_notifications
        self.transfer_notifications = transfer_notifications
        
    def __str__(self):
        return f"TokenTracker(policy_id={self.policy_id}, token_name={self.token_name}, channel_id={self.channel_id})"

    def increment_trade_notifications(self):
        """Increment the trade notifications counter"""
        self.trade_notifications += 1

    def increment_transfer_notifications(self):
        """Increment the transfer notifications counter"""
        self.transfer_notifications += 1

async def send_transaction_notification(tracker, tx_type, ada_amount, token_amount, details):
    """Send a notification about a transaction to the appropriate Discord channel"""
    try:
        # Check if the token amount meets the threshold
        if token_amount < tracker.threshold:
            logger.info(f"Token amount {token_amount} below threshold {tracker.threshold}, skipping notification")
            return
            
        # For transfers, check if we should track them
        if tx_type == 'wallet_transfer' and not tracker.track_transfers:
            logger.info("Transfer tracking disabled, skipping notification")
            return

        channel = bot.get_channel(tracker.channel_id)
        if not channel:
            logger.error(f"Could not find channel {tracker.channel_id}")
            return
            
        # Get token info
        token_info = await get_token_info(tracker.policy_id)
        token_name = token_info.get('name', 'Unknown Token')
        
        # Create embed
        embed = discord.Embed(
            title=f"{token_name} Transaction Detected!",
            description=(
                f"Transaction Hash: [`{details.get('hash', '')[:8]}...{details.get('hash', '')[-8:]}`](https://pool.pm/tx/{details.get('hash', '')})\n"
                f"Block Height: `{details.get('block_height', '')}`"
            ),
            color=discord.Color.blue()
        )
        
        # Add transaction details
        if tx_type == 'dex_trade':
            embed.add_field(
                name="Transaction Type",
                value="DEX Trade",
                inline=False
            )
        elif tx_type == 'wallet_transfer':
            embed.add_field(
                name="Transaction Type",
                value="Wallet Transfer",
                inline=False
            )
        
        # Add amounts
        if ada_amount:
            embed.add_field(
                name="ADA Amount",
                value=f"{ada_amount:,.2f} ₳",
                inline=True
            )
        if token_amount:
            embed.add_field(
                name="Token Amount",
                value=f"{token_amount:,}",
                inline=True
            )
            
        # Add any additional details
        if details:
            for key, value in details.items():
                if key != 'error':  # Don't show error details in Discord
                    embed.add_field(
                        name=key.replace('_', ' ').title(),
                        value=str(value),
                        inline=False
                    )
        
        # Send the notification
        await channel.send(embed=embed)
        
        # Increment notification counter
        if tx_type == 'dex_trade':
            tracker.increment_trade_notifications()
        elif tx_type == 'wallet_transfer':
            tracker.increment_transfer_notifications()
        
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}", exc_info=True)

def get_token_info(api: BlockFrostApi, policy_id: str):
    try:
        # Get all assets under this policy
        assets = api.assets_policy(policy_id)
        if isinstance(assets, Exception):
            raise assets
        return assets[0] if assets else None
    except Exception as e:
        logger.error(f"Error getting token info: {str(e)}", exc_info=True)
        return None

def get_transaction_details(api: BlockFrostApi, tx_hash: str):
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

def analyze_transaction_improved(tx_details, policy_id):
    """
    Enhanced transaction analysis that detects DEX trades by analyzing transaction patterns
    Returns: (type, ada_amount, token_amount, details)
    """
    try:
        # Get the utxos
        utxos = tx_details.get('utxos', {})
        if not utxos:
            logger.warning(f"No UTXOs found in transaction")
            return 'unknown', 0, 0, {}

        # Initialize amounts
        ada_in = 0
        ada_out = 0
        token_in = 0
        token_out = 0
        details = {}

        # Check inputs
        for utxo in utxos.get('inputs', []):
            if 'amount' in utxo:
                for amount in utxo['amount']:
                    if 'unit' in amount and policy_id in amount['unit']:
                        token_in += int(amount['quantity'])
                    elif 'unit' in amount and amount['unit'] == 'lovelace':
                        ada_in += int(amount['quantity'])

        # Check outputs
        for utxo in utxos.get('outputs', []):
            if 'amount' in utxo:
                for amount in utxo['amount']:
                    if 'unit' in amount and policy_id in amount['unit']:
                        token_out += int(amount['quantity'])
                    elif 'unit' in amount and amount['unit'] == 'lovelace':
                        ada_out += int(amount['quantity'])

        # Convert lovelace to ADA
        ada_in = ada_in / 1_000_000
        ada_out = ada_out / 1_000_000
        
        # Calculate net amounts
        net_ada = ada_out - ada_in
        net_tokens = token_out - token_in

        # Determine transaction type
        MIN_ADA_FOR_TRADE = 3  # Minimum ADA difference to consider it a trade
        
        if abs(net_tokens) < 100:  # Very small token movement, probably just fees
            return 'unknown', abs(net_ada), abs(net_tokens), details
            
        if token_in > 0 and token_out > 0:
            if abs(token_in - token_out) < token_in * 0.01:  # Less than 1% difference
                return 'wallet_transfer', abs(net_ada), max(token_in, token_out), details
                
        # If significant ADA movement, it's likely a trade
        if abs(net_ada) > MIN_ADA_FOR_TRADE:
            details['direction'] = 'buy' if net_tokens > 0 else 'sell'
            details['price_per_token'] = abs(net_ada / net_tokens) if net_tokens != 0 else 0
            return 'dex_trade', abs(net_ada), abs(net_tokens), details
            
        # If we see tokens but minimal ADA, it's probably a transfer
        return 'wallet_transfer', abs(net_ada), max(token_in, token_out), details

    except Exception as e:
        logger.error(f"Error in analyze_transaction_improved: {str(e)}", exc_info=True)
        return 'unknown', 0, 0, {}

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
                for amount in tx_details.get('outputs', []):
                    if amount.get('address') == addr:
                        for token in amount.get('amount', []):
                            if token.get('unit', '').startswith(policy_id) and int(token.get('quantity', 0)) > 0:
                                main_wallet = addr
                                break
        else:
            for addr in input_addresses:
                for amount in tx_details.get('inputs', []):
                    if amount.get('address') == addr:
                        for token in amount.get('amount', []):
                            if token.get('unit', '').startswith(policy_id) and int(token.get('quantity', 0)) > 0:
                                main_wallet = addr
                                break
        
        embed = discord.Embed(
            title=f"{title_emoji} Token {action_word} Detected",
            description=(
                f"Transaction Hash: [`{tx_details.get('hash', '')[:8]}...{tx_details.get('hash', '')[-8:]}`](https://pool.pm/tx/{tx_details.get('hash', '')})\n"
                f"Main Wallet: {create_pool_pm_link(main_wallet) if main_wallet else 'Unknown'}"
            ),
            color=discord.Color.green() if trade_type == "buy" else discord.Color.blue()
        )

        # Left side: Transaction Overview
        overview = (
            "```\n"
            f"Type     : DEX {action_word}\n"
            f"Block    : {tx_details.get('block', '')}\n"
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
        from_address = tx_details.get('inputs', [{}])[0].get('address', '')
        to_address = tx_details.get('outputs', [{}])[0].get('address', '')
        
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
            value=f"[View on CardanoScan](https://cardanoscan.io/transaction/{tx_details.get('hash', '')})",
            inline=False
        )

        # Set metadata
        embed.set_thumbnail(url=tracker.image_url)
        embed.timestamp = discord.utils.utcnow()
        embed.set_footer(
            text=f"Transfer detected at • Block #{tx_details.get('block_height', '')}",
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
                    database.delete_token_tracker(self.policy_id, interaction.channel_id)
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
                    database.save_token_tracker({
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
                    database.delete_token_tracker(policy_id, old_tracker.channel_id)
                except Exception as e:
                    logger.error(f"Failed to delete old token tracker from database: {str(e)}", exc_info=True)
                logger.info(f"Removed existing tracker for {policy_id}")

            # Create new tracker
            tracker = TokenTracker(
                policy_id=policy_id,
                token_name=token_name,
                image_url=image_url,
                threshold=threshold,
                channel_id=interaction.channel_id,
                track_transfers=track_transfers
            )
            
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
                    "• Real-time Notifications\n"
                    "• Customizable Thresholds"
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
                database.save_token_tracker({
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

        # Send the embed without any view/buttons
        await interaction.response.send_message(embed=embed)

    except Exception as e:
        logger.error(f"Error in status command: {str(e)}", exc_info=True)
        await interaction.response.send_message("Failed to get status. Please try again.", ephemeral=True)

@bot.tree.command(name="stop", description="Stop tracking all tokens in this channel")
async def stop(interaction: discord.Interaction):
    """Stop tracking all tokens in this channel"""
    try:
        # Check if there are any trackers for this channel
        channel_trackers = [t for t in active_trackers.values() if t.channel_id == interaction.channel_id]
        if not channel_trackers:
            await interaction.response.send_message("No tokens are being tracked in this channel.", ephemeral=True)
            return

        # Create confirmation embed
        embed = discord.Embed(
            title="⚠️ Stop Token Tracking",
            description="Are you sure you want to stop tracking all tokens in this channel?\nThis action cannot be undone.",
            color=discord.Color.yellow()
        )

        # List tokens that will be removed
        tokens_list = "\n".join([f"• {t.token_name} (`{t.policy_id}`)" for t in channel_trackers])
        embed.add_field(name="Tokens to remove:", value=tokens_list, inline=False)

        # Create confirmation buttons
        class ConfirmView(discord.ui.View):
            def __init__(self):
                super().__init__(timeout=60)  # 60 second timeout

            @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
            async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
                """Stop tracking the token"""
                try:
                    # Remove from database
                    database.remove_all_trackers_for_channel(interaction.channel_id)

                    # Remove from active trackers
                    policies_to_remove = []
                    for policy_id, tracker in active_trackers.items():
                        if tracker.channel_id == interaction.channel_id:
                            policies_to_remove.append(policy_id)
                    
                    for policy_id in policies_to_remove:
                        del active_trackers[policy_id]

                    # Disable buttons
                    for child in self.children:
                        child.disabled = True
                    
                    # Update message
                    embed = discord.Embed(
                        title="✅ Token Tracking Stopped",
                        description="Successfully stopped tracking all tokens in this channel.",
                        color=discord.Color.green()
                    )
                    await interaction.response.edit_message(embed=embed, view=self)

                except Exception as e:
                    logger.error(f"Error stopping token tracking: {str(e)}", exc_info=True)
                    await interaction.response.send_message("Failed to stop token tracking. Please try again.", ephemeral=True)

            @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
            async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
                # Disable buttons
                for child in self.children:
                    child.disabled = True
                
                # Update message
                embed = discord.Embed(
                    title="❌ Operation Cancelled",
                    description="Token tracking will continue.",
                    color=discord.Color.red()
                )
                await interaction.response.edit_message(embed=embed, view=self)

        # Send confirmation message
        await interaction.response.send_message(embed=embed, view=ConfirmView())

    except Exception as e:
        logger.error(f"Error in stop command: {str(e)}", exc_info=True)
        await interaction.response.send_message("Failed to process stop command. Please try again.", ephemeral=True)

# Run the bot
if __name__ == "__main__":
    bot.run(os.getenv('DISCORD_TOKEN'))
