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
                
            # Get all trackers and check each transaction against them
            trackers = database.get_trackers()
            logger.info(f"Checking transaction against {len(trackers)} tracked tokens")
            
            for tracker in trackers:
                try:
                    # Create composite key from policy_id and channel_id
                    tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
                    
                    # Skip if tracker not in memory (shouldn't happen)
                    if tracker_key not in active_trackers:
                        logger.warning(f"Tracker {tracker_key} not found in memory, skipping")
                        continue
                        
                    # Get active tracker from memory
                    active_tracker = active_trackers[tracker_key]
                    
                    # Process transaction for this tracker
                    await process_transaction_for_tracker(active_tracker, tx, inputs, outputs, tx_hash)
                    
                except Exception as e:
                    logger.error(f"Error processing transaction for tracker {tracker_key}: {str(e)}", exc_info=True)
                    continue
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_transaction_for_tracker(active_tracker, tx, inputs, outputs, tx_hash):
    """Process a transaction for a specific token tracker"""
    # Check if channel is accessible
    channel = bot.get_channel(active_tracker.channel_id)
    if not channel:
        for guild in bot.guilds:
            channel = guild.get_channel(active_tracker.channel_id)
            if channel:
                break
    
    if not channel:
        logger.error(f"Channel {active_tracker.channel_id} not found in any guild")
        # Remove tracker since channel is not accessible
        tracker_key = f"{active_tracker.policy_id}:{active_tracker.channel_id}"
        del active_trackers[tracker_key]
        database.delete_token_tracker(active_tracker.policy_id, active_tracker.channel_id)
        return
    
    # Check if any of our tracked tokens are involved
    is_involved = False
    
    # Check inputs and outputs for our policy ID
    input_addresses = []
    for inp in inputs:
        input_addresses.append(inp.get('address', ''))
        for amt in inp.get('amount', []):
            unit = amt.get('unit', '')
            # Debug log the unit we're checking
            logger.debug(f"Checking input unit: {unit}")
            
            # Only check policy ID in webhook handler
            if unit.startswith(active_tracker.policy_id):
                is_involved = True
                logger.info(f"Found token {active_tracker.token_name} in transaction inputs")
                break
        if is_involved:
            break
        
    # Check outputs if not found in inputs
    if not is_involved:
        output_addresses = []
        for out in outputs:
            output_addresses.append(out.get('address', ''))
            for amt in out.get('amount', []):
                unit = amt.get('unit', '')
                # Debug log the unit we're checking
                logger.debug(f"Checking output unit: {unit}")
                
                # Only check policy ID in webhook handler
                if unit.startswith(active_tracker.policy_id):
                    is_involved = True
                    logger.info(f"Found token {active_tracker.token_name} in transaction outputs")
                    break
            if is_involved:
                break
    
    if is_involved:
        # Log token involvement
        logger.info(f"Analyzing transaction for {active_tracker.token_name} ({active_tracker.policy_id})")
        
        # Create transaction data structure for analysis
        analysis_data = {
            'inputs': inputs,
            'outputs': outputs,
            'tx': tx
        }
        
        # Analyze the transaction
        tx_type, ada_amount, token_amount, details = analyze_transaction_improved(analysis_data, active_tracker.policy_id)
        
        # Add transaction hash to details
        details['hash'] = tx_hash
        
        # Log analysis results
        logger.info(f"Analysis results: type={tx_type}, ADA={ada_amount:.2f}, Tokens={token_amount:.2f}")
        
        # Send notification
        await send_transaction_notification(active_tracker, tx_type, ada_amount, token_amount, details)
    else:
        logger.debug(f"Token {active_tracker.token_name} not involved in transaction {tx_hash}")

def run_webhook_server():
    """Run the FastAPI webhook server"""
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))

@bot.event
async def on_ready():
    """Called when the bot is ready"""
    try:
        # Load all trackers from database
        trackers = database.get_trackers()
        logger.info(f"Loading {len(trackers)} trackers from database")
        
        for tracker in trackers:
            try:
                # Create composite key from policy_id and channel_id
                tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
                
                # Convert database model to TokenTracker object
                active_trackers[tracker_key] = TokenTracker(
                    policy_id=tracker.policy_id,
                    channel_id=int(tracker.channel_id),  # Ensure it's an int
                    token_name=tracker.token_name,
                    image_url=tracker.image_url,
                    threshold=tracker.threshold,
                    track_transfers=tracker.track_transfers,
                    last_block=tracker.last_block,
                    trade_notifications=tracker.trade_notifications,
                    transfer_notifications=tracker.transfer_notifications,
                    token_info=tracker.token_info
                )
                
                # Check if channel is accessible
                channel = bot.get_channel(tracker.channel_id)
                if not channel:
                    for guild in bot.guilds:
                        channel = guild.get_channel(tracker.channel_id)
                        if channel:
                            break
                
                if not channel:
                    logger.error(f"Channel {tracker.channel_id} not found in any guild")
                    # Remove tracker since channel is not accessible
                    del active_trackers[tracker_key]
                    database.delete_token_tracker(tracker.policy_id, tracker.channel_id)
                    continue
                
                logger.info(f"Loaded tracker for {tracker.token_name} in channel {tracker.channel_id}")
            except Exception as e:
                logger.error(f"Error loading tracker {tracker.policy_id}: {str(e)}", exc_info=True)
                
        # Sync slash commands
        await bot.tree.sync()
        logger.info(f"Bot is ready! Loaded {len(active_trackers)} trackers")
        
        # Log guild information
        for guild in bot.guilds:
            logger.info(f"Connected to guild: {guild.name} (ID: {guild.id})")
            logger.info(f"Available channels: {[f'{c.name} (ID: {c.id})' for c in guild.channels]}")
            
    except Exception as e:
        logger.error(f"Error in on_ready: {str(e)}", exc_info=True)

# Store active tracking configurations
active_trackers = {}

class TokenTracker:
    def __init__(self, policy_id: str, channel_id: int, token_name: str = None, 
                 image_url: str = None, threshold: float = 1000.0, 
                 track_transfers: bool = True, last_block: int = None,
                 trade_notifications: int = 0, transfer_notifications: int = 0,
                 token_info: dict = None):
        self.policy_id = policy_id
        self.channel_id = channel_id
        self.token_name = token_name
        self.image_url = image_url
        self.threshold = threshold
        self.track_transfers = track_transfers
        self.last_block = last_block
        self.trade_notifications = trade_notifications
        self.transfer_notifications = transfer_notifications
        
        # Get token info including decimals if not provided
        self.token_info = token_info or get_token_info(policy_id)
        if self.token_info:
            logger.info(f"Token {token_name} has {self.token_info.get('decimals', 0)} decimals")
        
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
        # For transfers, check if we should track them
        if tx_type == 'wallet_transfer' and not tracker.track_transfers:
            logger.info("Transfer tracking disabled, skipping notification")
            return

        # Get decimals for threshold comparison
        decimals = tracker.token_info.get('decimals', 0) if tracker.token_info else 0
        human_readable_amount = token_amount
        
        # Check if the token amount meets the threshold
        if human_readable_amount < tracker.threshold:
            logger.info(f"Token amount {human_readable_amount:,.{decimals}f} below threshold {tracker.threshold}, skipping notification")
            return
            
        # Try to get the channel from cache first
        channel = bot.get_channel(tracker.channel_id)
        
        # If not in cache, try to find it in guilds
        if not channel:
            for guild in bot.guilds:
                channel = guild.get_channel(tracker.channel_id)
                if channel:
                    break
                    
        if not channel:
            logger.error(f"Channel {tracker.channel_id} not found in any guild")
            # Remove tracker since channel is not accessible
            tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
            if tracker_key in active_trackers:
                del active_trackers[tracker_key]
            database.delete_token_tracker(tracker.policy_id, tracker.channel_id)
            return
            
        # Create appropriate embed based on transaction type
        if tx_type in ['buy', 'sell']:
            # Create trade embed
            embed = await create_trade_embed(details, tracker.policy_id, ada_amount, token_amount, tracker, details)
            if embed:
                try:
                    await channel.send(embed=embed)
                    tracker.increment_trade_notifications()
                except discord.Forbidden:
                    logger.error(f"Bot does not have permission to send messages in channel {tracker.channel_id}")
                except Exception as e:
                    logger.error(f"Error sending trade notification to channel {tracker.channel_id}: {str(e)}")
        elif tx_type == 'wallet_transfer' and tracker.track_transfers:
            # Create transfer embed
            embed = await create_transfer_embed(details, tracker.policy_id, token_amount, tracker)
            if embed:
                try:
                    await channel.send(embed=embed)
                    tracker.increment_transfer_notifications()
                except discord.Forbidden:
                    logger.error(f"Bot does not have permission to send messages in channel {tracker.channel_id}")
                except Exception as e:
                    logger.error(f"Error sending transfer notification to channel {tracker.channel_id}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error sending transaction notification: {str(e)}", exc_info=True)

def get_token_info(policy_id: str):
    """Get token information including metadata and decimals"""
    try:
        # Get all assets under this policy
        assets = api.assets_policy(policy_id)
        if isinstance(assets, Exception):
            raise assets
            
        if not assets:
            return None
            
        # Get the first asset (main token)
        asset = assets[0]
        
        # Get detailed metadata for the asset
        metadata = api.asset(asset.asset)
        if isinstance(metadata, Exception):
            raise metadata
            
        # Convert Namespace objects to dictionaries
        def namespace_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return [namespace_to_dict(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: namespace_to_dict(v) for k, v in obj.items()}
            else:
                return obj
                
        # Convert metadata to dictionary
        metadata_dict = namespace_to_dict(metadata)
        
        # Get onchain metadata if available
        onchain_metadata = metadata_dict.get('onchain_metadata', {})
        
        # Try to get decimals from various sources
        decimals = None
        
        # Check onchain metadata first (CIP-67 standard)
        if onchain_metadata and isinstance(onchain_metadata, dict):
            decimals = onchain_metadata.get('decimals')
            
        # If not found, check asset metadata
        if decimals is None and 'metadata' in metadata_dict:
            decimals = metadata_dict['metadata'].get('decimals')
            
        # Default to 0 if no decimal information found
        if decimals is None:
            decimals = 0
            logger.info(f"No decimal information found for {asset.asset}, defaulting to 0")
        else:
            logger.info(f"Found {decimals} decimals for {asset.asset}")
            
        return {
            'asset': asset.asset,
            'policy_id': policy_id,
            'name': metadata_dict.get('asset_name'),
            'decimals': int(decimals),
            'metadata': onchain_metadata
        }
        
    except Exception as e:
        logger.error(f"Error getting token info: {str(e)}", exc_info=True)
        return None

def format_token_amount(amount: int, decimals: int) -> str:
    """Format token amount considering decimals"""
    if decimals == 0:
        return f"{amount:,}"
    
    # Convert to float and divide by 10^decimals
    formatted = amount / (10 ** decimals)
    
    # Format with appropriate decimal places
    if decimals <= 2:
        return f"{formatted:,.{decimals}f}"
    elif formatted >= 1000:
        return f"{formatted:,.2f}"
    else:
        return f"{formatted:,.{min(decimals, 6)}f}"

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
        # Get the inputs and outputs
        inputs = tx_details.get('inputs', [])
        outputs = tx_details.get('outputs', [])
        if not inputs and not outputs:
            logger.warning(f"No inputs/outputs found in transaction")
            return 'unknown', 0, 0, {}

        # Initialize amounts
        ada_in = 0
        ada_out = 0
        token_in = 0
        token_out = 0
        details = {}
        
        # Get token info for decimal handling
        token_info = get_token_info(policy_id)
        decimals = token_info.get('decimals', 0) if token_info else 0
        logger.info(f"Found {decimals} decimals for {policy_id}")

        # Construct full asset name (policy_id + hex of token name)
        full_asset_name = None
        if token_info and 'name' in token_info:
            # The asset name from token_info is already hex-encoded
            full_asset_name = f"{policy_id}{token_info['name']}"
            logger.info(f"Looking for full asset name: {full_asset_name}")

        # Check inputs
        input_addresses = []
        for inp in inputs:
            input_addresses.append(inp.get('address', ''))
            for amount in inp.get('amount', []):
                unit = amount.get('unit', '')
                # Debug log the unit we're checking
                logger.debug(f"Checking input unit: {unit}")
                
                # Compare with full asset name or policy ID
                if unit == full_asset_name or unit.startswith(policy_id):
                    raw_amount = int(amount['quantity'])
                    token_in += raw_amount
                    logger.info(f"Found {raw_amount} tokens in input with unit {unit}")
                elif unit == 'lovelace':
                    ada_in += int(amount['quantity'])

        # Check outputs
        output_addresses = []
        for out in outputs:
            output_addresses.append(out.get('address', ''))
            for amount in out.get('amount', []):
                unit = amount.get('unit', '')
                # Debug log the unit we're checking
                logger.debug(f"Checking output unit: {unit}")
                
                # Compare with full asset name or policy ID
                if unit == full_asset_name or unit.startswith(policy_id):
                    raw_amount = int(amount['quantity'])
                    token_out += raw_amount
                    logger.info(f"Found {raw_amount} tokens in output with unit {unit}")
                elif unit == 'lovelace':
                    ada_out += int(amount['quantity'])

        # Convert lovelace to ADA
        ada_in = ada_in / 1_000_000
        ada_out = ada_out / 1_000_000

        # Calculate net amounts
        ada_amount = abs(ada_out - ada_in)
        
        # For wallet transfers, use the largest output amount
        # For buys/sells, use the difference between in and out
        if token_in > 0 and token_out > 0:
            # Find the largest single output amount
            max_output = 0
            for out in outputs:
                for amount in out.get('amount', []):
                    unit = amount.get('unit', '')
                    if unit == full_asset_name or unit.startswith(policy_id):
                        output_amount = int(amount['quantity'])
                        max_output = max(max_output, output_amount)
            
            raw_token_amount = max_output
            logger.info(f"Wallet transfer - using largest output amount: {raw_token_amount}")
        else:
            # Buy/Sell - use the difference
            raw_token_amount = abs(token_out - token_in)
            logger.info(f"Buy/Sell - using difference: {raw_token_amount}")
        
        # Only apply decimal conversion if decimals > 0
        # For tokens with 0 decimals, use the raw amount
        token_amount = raw_token_amount / (10 ** decimals) if decimals > 0 else raw_token_amount
        logger.info(f"Raw token amount: {raw_token_amount}, Decimals: {decimals}, Converted amount: {token_amount}")

        # Log token movement for debugging
        logger.info(f"Token input: {token_in}, Token output: {token_out}")
        logger.info(f"ADA input: {ada_in}, ADA output: {ada_out}")

        # Store details for notification
        details = {
            'ada_in': ada_in,
            'ada_out': ada_out,
            'token_in': token_in,
            'token_out': token_out,
            'raw_token_amount': raw_token_amount,
            'decimals': decimals,
            'full_asset_name': full_asset_name
        }

        # Determine transaction type
        if token_in > 0 and token_out > 0:
            return 'wallet_transfer', ada_amount, token_amount, details
        elif token_in > 0 and token_out == 0:
            return 'sell', ada_amount, token_amount, details
        elif token_in == 0 and token_out > 0:
            return 'buy', ada_amount, token_amount, details
        else:
            return 'unknown', 0, 0, details

    except Exception as e:
        logger.error(f"Error analyzing transaction: {str(e)}", exc_info=True)
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
                            if token.get('unit') == full_asset_name:
                                main_wallet = addr
                                break
        else:
            for addr in input_addresses:
                for amount in tx_details.get('inputs', []):
                    if amount.get('address') == addr:
                        for token in amount.get('amount', []):
                            if token.get('unit') == full_asset_name:
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
                f"Tokens Recv: {format_token_amount(int(ada_amount * 10**tracker.token_info.get('decimals', 0)), tracker.token_info.get('decimals', 0))}\n"
                f"Price/Token: {(ada_amount/token_amount):.6f}\n"
                "```"
            )
        else:
            trade_info = (
                "```\n"
                f"Tokens Sold: {format_token_amount(int(token_amount * 10**tracker.token_info.get('decimals', 0)), tracker.token_info.get('decimals', 0))}\n"
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
            f"**Amount:** ```{format_token_amount(int(token_amount * 10**tracker.token_info.get('decimals', 0)), tracker.token_info.get('decimals', 0))} Tokens```"
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
    def __init__(self, policy_id):
        super().__init__(timeout=None)  # No timeout for persistent buttons
        self.policy_id = policy_id

    @discord.ui.button(label="Stop Tracking", style=discord.ButtonStyle.danger, emoji="⛔")
    async def stop_tracking(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            tracker_key = f"{self.policy_id}:{interaction.channel_id}"
            if tracker_key in active_trackers:
                # Remove from database
                database.delete_token_tracker(self.policy_id, interaction.channel_id)
                
                # Remove from active trackers
                del active_trackers[tracker_key]
            
                # Update message
                embed = discord.Embed(
                    title="✅ Token Tracking Stopped",
                    description=f"Successfully stopped tracking token with policy ID: ```{self.policy_id}```",
                    color=discord.Color.green()
                )
                
                # Disable all buttons
                for child in self.children:
                    child.disabled = True
                
                await interaction.response.edit_message(embed=embed, view=self)
                logger.info(f"Stopped tracking token: {self.policy_id}")
            else:
                await interaction.response.send_message("Not tracking this token anymore.", ephemeral=True)
                
        except Exception as e:
            logger.error(f"Error stopping token tracking: {str(e)}", exc_info=True)
            await interaction.response.send_message("Failed to stop token tracking. Please try again.", ephemeral=True)

    @discord.ui.button(label="Toggle Transfers", style=discord.ButtonStyle.primary, emoji="🔄")
    async def toggle_transfers(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            # Get the tracker
            tracker_key = f"{self.policy_id}:{interaction.channel_id}"
            tracker = active_trackers.get(tracker_key)
            if not tracker:
                await interaction.response.send_message("❌ Token tracker not found.", ephemeral=True)
                return

            # Toggle transfer notifications
            tracker.track_transfers = not tracker.track_transfers
            
            # Update in database
            database.save_token_tracker({
                'policy_id': self.policy_id,
                'token_name': tracker.token_name,
                'image_url': tracker.image_url,
                'threshold': tracker.threshold,
                'channel_id': interaction.channel_id,
                'last_block': tracker.last_block,
                'track_transfers': tracker.track_transfers,
                'token_info': tracker.token_info
            })

            # Create updated embed
            embed = discord.Embed(
                title="✅ Token Tracking Active",
                description="Currently tracking the following token:",
                color=discord.Color.blue()
            )
            
            # Basic token info
            token_text = (
                f"**Policy ID:** ```{tracker.policy_id}```\n"
                f"**Name:** ```{tracker.token_name}```"
            )
            embed.add_field(
                name="Token Information",
                value=token_text,
                inline=False
            )
            
            # Configuration section
            config_text = (
                f"**Threshold:** ```{tracker.threshold:,.2f} Tokens```\n"
                f"**Transfer Notifications:** ```{'Enabled' if tracker.track_transfers else 'Disabled'}```\n"
            )
            embed.add_field(
                name="",
                value=config_text,
                inline=False
            )

            # Statistics
            stats_text = (
                f"**Trade Notifications:** ```{tracker.trade_notifications}```\n"
                f"**Transfer Notifications:** ```{tracker.transfer_notifications}```\n"
            )
            embed.add_field(
                name="",
                value=stats_text,
                inline=False
            )

            # Set token image if available
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)

            await interaction.response.edit_message(embed=embed)
            logger.info(f"Toggled transfer notifications for {tracker.token_name} to {tracker.track_transfers}")
            
        except Exception as e:
            logger.error(f"Error toggling transfers: {str(e)}", exc_info=True)
            await interaction.response.send_message("❌ Failed to toggle transfer notifications.", ephemeral=True)

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
        """Handle form submission"""
        try:
            # Get token info first to validate the policy ID
            token_info = get_token_info(self.policy_id.value)
            if not token_info:
                await interaction.response.send_message(
                    "❌ Could not find token with that policy ID. Please check the ID and try again.",
                    ephemeral=True
                )
                return
                
            # Parse threshold
            try:
                threshold = float(self.threshold.value) if self.threshold.value else 1000.0
            except ValueError:
                await interaction.response.send_message(
                    "❌ Invalid threshold value. Please enter a valid number.",
                    ephemeral=True
                )
                return
                
            # Parse track_transfers
            track_transfers = self.track_transfers.value.lower() != 'no'
            
            # Create tracker
            tracker = TokenTracker(
                policy_id=self.policy_id.value,
                channel_id=interaction.channel_id,
                token_name=self.token_name.value,
                image_url=self.image_url.value if self.image_url.value else None,
                threshold=threshold,
                track_transfers=track_transfers,
                token_info=token_info
            )
            
            # Add to database
            database.add_tracker(
                policy_id=tracker.policy_id,
                channel_id=tracker.channel_id,
                token_name=tracker.token_name,
                image_url=tracker.image_url,
                threshold=tracker.threshold,
                track_transfers=tracker.track_transfers,
                token_info=tracker.token_info
            )
            
            # Create success embed
            embed = discord.Embed(
                title="Token Tracking Started",
                description="Successfully initialized tracking for:",
                color=discord.Color.blue()
            )
            
            # Basic token info
            embed.add_field(
                name="Token",
                value=f"```{tracker.token_name}```",
                inline=True
            )
            embed.add_field(
                name="Policy ID",
                value=f"```{tracker.policy_id}```",
                inline=False
            )
            
            # Configuration section
            config_text = (
                f"**Threshold:** ```{threshold:,.2f} Tokens```\n"

                f"**Transfer Notifications:** ```{'Enabled' if track_transfers else 'Disabled'}```\n"

            )
            embed.add_field(
                name="",
                value=config_text,
                inline=False
            )

            # Statistics
            stats_text = (
                f"**Trade Notifications:** ```0```\n"
                f"**Transfer Notifications:** ```0```\n"
            )
            embed.add_field(
                name="",
                value=stats_text,
                inline=False
            )

            # Set token image if available
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)


            
            # Create view with buttons
            view = TokenControls(tracker.policy_id)
            
            # Update the loading message
            await interaction.response.send_message(embed=embed, view=view)
            
            # Save to memory and database
            tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
            active_trackers[tracker_key] = tracker
            
            logger.info(f"Started tracking token: {tracker.token_name} ({tracker.policy_id})")
                
        except Exception as e:
            logger.error(f"Error in token setup: {str(e)}", exc_info=True)
            error_embed = discord.Embed(
                title="❌ Error",
                description="Failed to start tracking. Please check your inputs and try again.",
                color=discord.Color.red()
            )
            await interaction.response.send_message(embed=error_embed)

async def send_start_message(interaction, policy_id, token_name, threshold, track_transfers, image_url):
    """Send the initial token tracking started message"""
    embed = discord.Embed(
        title="Token Tracking Active",
        description="Successfully initialized tracking for:",
        color=discord.Color.blue()
    )

    # Basic token info
    embed.add_field(
        name="Token",
        value=f"```{token_name}```",
        inline=True
    )
    embed.add_field(
        name="Policy ID",
        value=f"```{policy_id}```",
        inline=False
    )

    # Configuration section
    config_text = (
        f"**Threshold:** ```{threshold:,.2f} Tokens```\n"

        f"**Transfer Notifications:** ```{'Enabled' if track_transfers else 'Disabled'}```\n"

    )
    embed.add_field(
        name="",
        value=config_text,
        inline=False
    )


    # Set token image if available
    if image_url:
        embed.set_thumbnail(url=image_url)

    # Add control buttons
    view = TokenControls(policy_id)
    await interaction.response.send_message(embed=embed, view=view)

@bot.tree.command(name="start", description="Start tracking token purchases and transfers")
async def start(interaction: discord.Interaction):
    """Start tracking a token by opening a setup form"""
    modal = TokenSetupModal()
    await interaction.response.send_modal(modal)

@bot.tree.command(name="help")
async def help_command(interaction: discord.Interaction):
    """Display help information about the bot's commands"""
    embed = discord.Embed(
        title="Help",
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
@app_commands.default_permissions(administrator=True)
async def status_command(interaction: discord.Interaction):
    """Show status of tracked tokens in this channel"""
    try:
        # Get all trackers for this channel
        channel_trackers = [t for t in active_trackers.values() if t.channel_id == interaction.channel_id]
        if not channel_trackers:
            embed = discord.Embed(
                title="❌ No Active Trackers",
                description="No tokens are currently being tracked in this channel.",
                color=discord.Color.red()
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        # Create an embed for each tracker
        for tracker in channel_trackers:
            embed = discord.Embed(
                title="Token Tracking Active",
                description="Currently tracking the following token:",
                color=discord.Color.blue()
            )
            
            # Basic token info
            embed.add_field(
                name="Token",
                value=f"```{tracker.token_name}```",
                inline=True
            )
            embed.add_field(
                name="Policy ID",
                value=f"```{tracker.policy_id}```",
                inline=False
            )
            
            # Configuration section
            config_text = (
                f"**Threshold:** ```{tracker.threshold:,.2f} Tokens```\n"
                f"**Transfer Notifications:** ```{'Enabled' if tracker.track_transfers else 'Disabled'}```\n"
            )
            embed.add_field(
                name="",
                value=config_text,
                inline=False
            )

            # Statistics
            stats_text = (
                f"**Sale Notifications:** ```{tracker.trade_notifications}```\n"
                f"**Transfer Notifications:** ```{tracker.transfer_notifications}```\n"
            )
            embed.add_field(
                name="",
                value=stats_text,
                inline=False
            )

            # Set token image if available
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)

            # Add control buttons
            view = TokenControls(tracker.policy_id)
            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    except Exception as e:
        logger.error(f"Error in status command: {str(e)}", exc_info=True)
        error_embed = discord.Embed(
            title="❌ Error",
            description="Failed to retrieve token tracking status.",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=error_embed, ephemeral=True)

@bot.tree.command(name="stop", description="Stop tracking all tokens in this channel")
@app_commands.default_permissions(administrator=True)
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
