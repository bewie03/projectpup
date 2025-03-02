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
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import hmac
import hashlib
import uvicorn
import threading
import time
from queue import Queue
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Configure logging with consistent style
logger = logging.getLogger('CardanoTracker')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

# SQLAlchemy Database Setup
Base = declarative_base()

class TokenTracker(Base):
    """Model for tracking Cardano token configurations"""
    __tablename__ = 'trackers'
    policy_id = Column(String, primary_key=True)
    channel_id = Column(BigInteger, primary_key=True)
    token_name = Column(String)
    image_url = Column(String)
    threshold = Column(Float, default=1000.0)
    track_transfers = Column(Boolean, default=True)
    last_block = Column(BigInteger, default=0)
    trade_notifications = Column(Integer, default=0)
    transfer_notifications = Column(Integer, default=0)
    decimals = Column(Integer, default=0)

    def to_dict(self):
        """Convert tracker to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class Database:
    """Handles database operations for token tracking"""
    def __init__(self, database_url=None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("No database URL provided")
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_trackers(self):
        """Retrieve all token trackers from the database"""
        with self.Session() as session:
            try:
                trackers = session.query(TokenTracker).all()
                return trackers
            except SQLAlchemyError as e:
                logger.error(f"Database error retrieving trackers: {str(e)}")
                return []

    def add_tracker(self, **kwargs):
        """Add or update a token tracker in the database"""
        with self.Session() as session:
            try:
                tracker = session.query(TokenTracker).filter_by(
                    policy_id=kwargs['policy_id'],
                    channel_id=kwargs['channel_id']
                ).first()
                if tracker:
                    for key, value in kwargs.items():
                        setattr(tracker, key, value)
                else:
                    tracker = TokenTracker(**kwargs)
                    session.add(tracker)
                session.commit()
                return tracker
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error adding tracker: {str(e)}")
                return None

    def delete_tracker(self, policy_id, channel_id):
        """Remove a specific token tracker"""
        with self.Session() as session:
            try:
                tracker = session.query(TokenTracker).filter_by(
                    policy_id=policy_id,
                    channel_id=channel_id
                ).first()
                if tracker:
                    session.delete(tracker)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error deleting tracker: {str(e)}")
                return False

    def remove_all_trackers_for_channel(self, channel_id):
        """Remove all trackers for a specific channel"""
        with self.Session() as session:
            try:
                trackers = session.query(TokenTracker).filter_by(channel_id=channel_id).delete()
                session.commit()
                return bool(trackers)
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error removing trackers: {str(e)}")
                return False

    def update_tracker_stats(self, policy_id, channel_id, notification_type):
        """Update notification counts for a tracker"""
        with self.Session() as session:
            try:
                tracker = session.query(TokenTracker).filter_by(
                    policy_id=policy_id,
                    channel_id=channel_id
                ).first()
                if tracker:
                    if notification_type == 'trade':
                        tracker.trade_notifications += 1
                    elif notification_type == 'transfer':
                        tracker.transfer_notifications += 1
                    session.commit()
                return bool(tracker)
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error updating stats: {str(e)}")
                return False

database = Database()

# FastAPI setup for webhooks
app = FastAPI()
notification_queue = Queue()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Blockfrost API
api = BlockFrostApi(
    project_id=os.getenv('BLOCKFROST_API_KEY'),
    base_url=ApiUrls.mainnet.value
)

WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
WEBHOOK_TOLERANCE_SECONDS = 600
active_trackers = {}

class TokenTrackerClass:
    """Class to manage active token tracking configurations"""
    def __init__(self, policy_id, channel_id, token_name, image_url, threshold, track_transfers, decimals):
        self.policy_id = policy_id
        self.channel_id = channel_id
        self.token_name = token_name
        self.image_url = image_url
        self.threshold = threshold
        self.track_transfers = track_transfers
        self.decimals = decimals
        self.trade_notifications = 0
        self.transfer_notifications = 0

    def increment_trade_notifications(self):
        self.trade_notifications += 1
        database.update_tracker_stats(self.policy_id, self.channel_id, 'trade')

    def increment_transfer_notifications(self):
        self.transfer_notifications += 1
        database.update_tracker_stats(self.policy_id, self.channel_id, 'transfer')

def verify_webhook_signature(payload: bytes, header: str, current_time: int) -> bool:
    """Verify the authenticity of Blockfrost webhook signatures"""
    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET not set, skipping verification")
        return True
    try:
        pairs = dict(pair.split('=') for pair in header.split(','))
        timestamp = pairs.get('t')
        signature = pairs.get('v1')
        if not timestamp or not signature:
            logger.error("Missing timestamp or signature in header")
            return False
        timestamp_diff = abs(current_time - int(timestamp))
        if timestamp_diff > WEBHOOK_TOLERANCE_SECONDS:
            logger.error(f"Webhook timestamp too old: {timestamp_diff} seconds")
            return False
        signature_payload = f"{timestamp}.{payload.decode('utf-8')}"
        computed = hmac.new(
            WEBHOOK_SECRET.encode(),
            signature_payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(computed, signature)
    except Exception as e:
        logger.error(f"Error verifying webhook signature: {str(e)}")
        return False

@app.post("/webhook/transaction")
async def transaction_webhook(request: Request):
    """Handle incoming transaction webhooks from Blockfrost"""
    try:
        logger.info("Webhook request received")
        signature = request.headers.get('Blockfrost-Signature')
        if not signature:
            raise HTTPException(status_code=400, detail="Missing signature header")
        
        payload = await request.body()
        current_time = int(time.time())
        if not verify_webhook_signature(payload, signature, current_time):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        data = await request.json()
        if not isinstance(data, dict) or data.get('type') != 'transaction':
            logger.error(f"Invalid webhook data format: {data.get('type', 'unknown')}")
            return {"status": "ignored"}
        
        transactions = data.get('payload', [])
        if not transactions:
            logger.info("No transactions in webhook")
            return {"status": "no transactions"}
        
        for tx_data in transactions:
            tx = tx_data.get('tx', {})
            tx_hash = tx.get('hash', 'unknown')
            logger.info(f"Processing transaction: {tx_hash}")
            
            inputs = tx_data.get('inputs', [])
            outputs = tx_data.get('outputs', [])
            if not inputs or not outputs:
                logger.warning(f"Skipping transaction {tx_hash} - Missing inputs/outputs")
                continue
            
            for tracker in database.get_trackers():
                tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
                if tracker_key not in active_trackers:
                    active_trackers[tracker_key] = TokenTrackerClass(
                        tracker.policy_id, tracker.channel_id, tracker.token_name,
                        tracker.image_url, tracker.threshold, tracker.track_transfers, tracker.decimals
                    )
                
                active_tracker = active_trackers[tracker_key]
                analysis = analyze_transaction(tx_data, active_tracker)
                if analysis:
                    tx_type, ada_amount, token_amount, details = analysis
                    notification_queue.put({
                        'tracker': active_tracker,
                        'tx_type': tx_type,
                        'ada_amount': ada_amount,
                        'token_amount': token_amount,
                        'details': details
                    })
                    logger.info(f"Queued notification for {active_tracker.token_name}")
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@tasks.loop(seconds=1)
async def process_notification_queue():
    """Process queued notifications and send to Discord"""
    while not notification_queue.empty():
        try:
            data = notification_queue.get_nowait()
            await send_transaction_notification(**data)
            notification_queue.task_done()
        except Exception as e:
            logger.error(f"Error processing notification: {str(e)}")

@bot.event
async def on_ready():
    """Handle bot startup and initialize trackers"""
    if not process_notification_queue.is_running():
        process_notification_queue.start()
    for tracker in database.get_trackers():
        tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
        active_trackers[tracker_key] = TokenTrackerClass(
            tracker.policy_id, tracker.channel_id, tracker.token_name,
            tracker.image_url, tracker.threshold, tracker.track_transfers, tracker.decimals
        )
    await bot.tree.sync()
    logger.info(f"Bot ready with {len(active_trackers)} trackers")

def get_token_info(policy_id: str):
    """Fetch token metadata including decimals from Blockfrost"""
    try:
        assets = api.assets_policy(policy_id)
        if isinstance(assets, Exception) or not assets:
            return None
        asset = assets[0]
        metadata = api.asset(asset.asset)
        if isinstance(metadata, Exception):
            return None
        decimals = metadata.onchain_metadata.get('decimals', 0) if metadata.onchain_metadata else 0
        return {'decimals': decimals, 'name': metadata.asset_name}
    except Exception as e:
        logger.error(f"Error getting token info: {str(e)}")
        return None

def format_token_amount(amount: int, decimals: int) -> str:
    """Format token amount with correct decimal places"""
    if decimals == 0:
        return f"{amount:,}"
    amount_float = amount / (10 ** decimals)
    return f"{amount_float:,.{min(decimals, 6)}f}"

def analyze_transaction(tx_data, tracker):
    """Analyze transaction to determine type and amounts"""
    try:
        inputs = tx_data.get('inputs', [])
        outputs = tx_data.get('outputs', [])
        tx = tx_data.get('tx', {})
        tx_hash = tx.get('hash', 'unknown')

        ada_in, ada_out = 0, 0
        token_in, token_out = 0, 0
        policy_id = tracker.policy_id
        full_asset = f"{policy_id}{tracker.token_name}" if tracker.token_name else policy_id

        # Calculate ADA and token movements
        for inp in inputs:
            for amt in inp.get('amount', []):
                if amt['unit'] == 'lovelace':
                    ada_in += int(amt['quantity'])
                elif amt['unit'].startswith(policy_id):
                    token_in += int(amt['quantity'])

        for out in outputs:
            for amt in out.get('amount', []):
                if amt['unit'] == 'lovelace':
                    ada_out += int(amt['quantity'])
                elif amt['unit'].startswith(policy_id):
                    token_out += int(amt['quantity'])

        ada_amount = (ada_out - ada_in) / 1_000_000  # Convert lovelace to ADA
        raw_token_amount = token_out - token_in
        token_amount = raw_token_amount / (10 ** tracker.decimals) if tracker.decimals > 0 else raw_token_amount

        # Determine transaction type
        if abs(ada_amount) > 1.0:  # Likely a DEX trade
            if ada_amount < 0:  # ADA spent, tokens received (purchase)
                return 'buy', abs(ada_amount), token_amount, {
                    'hash': tx_hash, 'inputs': inputs, 'outputs': outputs
                }
            else:  # Tokens spent, ADA received (sale)
                return 'sell', ada_amount, abs(token_amount), {
                    'hash': tx_hash, 'inputs': inputs, 'outputs': outputs
                }
        elif token_in > 0 and token_out > 0:  # Token transfer
            return 'transfer', 0, abs(token_amount), {
                'hash': tx_hash, 'inputs': inputs, 'outputs': outputs
            }
        return None
    except Exception as e:
        logger.error(f"Error analyzing transaction: {str(e)}")
        return None

async def create_transfer_embed(tracker, token_amount, details):
    """Create an embed for token transfer notifications"""
    tx_hash = details['hash']
    inputs = details['inputs']
    outputs = details['outputs']

    sender = next((inp['address'] for inp in inputs if any(amt['unit'].startswith(tracker.policy_id) for amt in inp.get('amount', []))), 'Unknown')
    receiver = next((out['address'] for out in outputs if any(amt['unit'].startswith(tracker.policy_id) for amt in out.get('amount', []))), 'Unknown')

    embed = discord.Embed(
        title=f"↔️ {tracker.token_name} Transfer Detected",
        description=f"[View on CardanoScan](https://cardanoscan.io/transaction/{tx_hash})",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="Transfer Details",
        value=f"**From:** [{sender[:8]}...](https://cardanoscan.io/address/{sender})\n"
              f"**To:** [{receiver[:8]}...](https://cardanoscan.io/address/{receiver})\n"
              f"**Amount:** {format_token_amount(int(token_amount * (10 ** tracker.decimals)), tracker.decimals)} {tracker.token_name}",
        inline=False
    )
    if tracker.image_url:
        embed.set_thumbnail(url=tracker.image_url)
    embed.timestamp = datetime.now()
    return embed

async def create_trade_embed(tracker, tx_type, ada_amount, token_amount, details):
    """Create an embed for DEX trade notifications (buy/sell)"""
    tx_hash = details['hash']
    title = f"💰 {tracker.token_name} {'Purchase' if tx_type == 'buy' else 'Sale'} Detected"
    color = discord.Color.green() if tx_type == 'buy' else discord.Color.blue()

    embed = discord.Embed(
        title=title,
        description=f"[View on CardanoScan](https://cardanoscan.io/transaction/{tx_hash})",
        color=color
    )
    embed.add_field(
        name="Overview",
        value=f"Type: DEX {'Purchase' if tx_type == 'buy' else 'Sale'}\n"
              f"Status: Confirmed",
        inline=True
    )
    embed.add_field(
        name="Trade Details",
        value=f"{'ADA Spent' if tx_type == 'buy' else 'ADA Received'}: {abs(ada_amount):,.2f} ADA\n"
              f"{'Tokens Received' if tx_type == 'buy' else 'Tokens Sold'}: {format_token_amount(int(abs(token_amount) * (10 ** tracker.decimals)), tracker.decimals)} {tracker.token_name}\n"
              f"Price/Token: {abs(ada_amount/token_amount):,.6f} ADA",
        inline=True
    )

    # Find and format input/output addresses
    input_addrs = [inp['address'] for inp in details['inputs'] if inp.get('address')]
    output_addrs = [out['address'] for out in details['outputs'] if out.get('address')]
    
    embed.add_field(
        name="Addresses",
        value=f"📥 Inputs: {', '.join([f'[{addr[:8]}...](https://cardanoscan.io/address/{addr})' for addr in input_addrs[:2]]) or 'None'}\n"
              f"📤 Outputs: {', '.join([f'[{addr[:8]}...](https://cardanoscan.io/address/{addr})' for addr in output_addrs[:2]]) or 'None'}",
        inline=False
    )
    if tracker.image_url:
        embed.set_thumbnail(url=tracker.image_url)
    embed.timestamp = datetime.now()
    return embed

async def send_transaction_notification(tracker, tx_type, ada_amount, token_amount, details):
    """Send transaction notifications to Discord channels"""
    if tx_type == 'transfer' and not tracker.track_transfers:
        logger.info("Transfer tracking disabled, skipping notification")
        return
    if token_amount < tracker.threshold:
        logger.info(f"Amount {token_amount} below threshold {tracker.threshold}, skipping")
        return

    channel = bot.get_channel(tracker.channel_id) or await bot.fetch_channel(tracker.channel_id)
    if not channel:
        logger.error(f"Channel {tracker.channel_id} not found, removing tracker")
        database.delete_tracker(tracker.policy_id, tracker.channel_id)
        if f"{tracker.policy_id}:{tracker.channel_id}" in active_trackers:
            del active_trackers[f"{tracker.policy_id}:{tracker.channel_id}"]
        return

    embed = await (create_transfer_embed if tx_type == 'transfer' else create_trade_embed)(
        tracker, token_amount, details if tx_type == 'transfer' else (tx_type, ada_amount, token_amount, details)
    )
    if embed:
        try:
            await channel.send(embed=embed)
            if tx_type == 'transfer':
                tracker.increment_transfer_notifications()
            else:
                tracker.increment_trade_notifications()
            logger.info(f"Sent {tx_type} notification for {tracker.token_name}")
        except discord.Forbidden:
            logger.error(f"No permission to send in channel {channel.name}")
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")

class TokenControls(discord.ui.View):
    """View for controlling token tracking actions"""
    def __init__(self, policy_id):
        super().__init__(timeout=None)
        self.policy_id = policy_id

    @discord.ui.button(label="Stop Tracking", style=discord.ButtonStyle.danger, emoji="⛔")
    async def stop_tracking(self, interaction: discord.Interaction, button: discord.ui.Button):
        if database.delete_tracker(self.policy_id, interaction.channel_id):
            tracker_key = f"{self.policy_id}:{interaction.channel_id}"
            if tracker_key in active_trackers:
                del active_trackers[tracker_key]
            embed = discord.Embed(
                title="✅ Tracking Stopped",
                description=f"Stopped tracking token with policy ID: ```{self.policy_id}```",
                color=discord.Color.green()
            )
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(embed=embed, view=self)
            logger.info(f"Stopped tracking token: {self.policy_id}")
        else:
            await interaction.response.send_message("Not tracking this token anymore.", ephemeral=True)

    @discord.ui.button(label="Toggle Transfers", style=discord.ButtonStyle.primary, emoji="🔄")
    async def toggle_transfers(self, interaction: discord.Interaction, button: discord.ui.Button):
        tracker_key = f"{self.policy_id}:{interaction.channel_id}"
        tracker = active_trackers.get(tracker_key)
        if not tracker:
            await interaction.response.send_message("Token tracker not found.", ephemeral=True)
            return
        tracker.track_transfers = not tracker.track_transfers
        database.add_tracker(**tracker.__dict__)
        embed = discord.Embed(
            title="✅ Token Tracking Updated",
            description=f"Transfers {'enabled' if tracker.track_transfers else 'disabled'} for {tracker.token_name}",
            color=discord.Color.blue()
        )
        if tracker.image_url:
            embed.set_thumbnail(url=tracker.image_url)
        await interaction.response.edit_message(embed=embed)
        logger.info(f"Toggled transfers for {tracker.token_name} to {tracker.track_transfers}")

class TokenSetupModal(discord.ui.Modal, title="🪙 Token Setup"):
    """Modal for setting up new token tracking"""
    def __init__(self):
        super().__init__()
        self.policy_id = discord.ui.TextInput(
            label="Policy ID",
            placeholder="Enter the token's 56-character policy ID",
            style=discord.TextStyle.short,
            required=True,
            min_length=56,
            max_length=56
        )
        self.token_name = discord.ui.TextInput(
            label="Token Name",
            placeboholder="Enter the token's name",
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
        for item in [self.policy_id, self.token_name, self.image_url, self.threshold, self.track_transfers]:
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        token_info = get_token_info(self.policy_id.value)
        if not token_info or 'decimals' not in token_info:
            await interaction.response.send_message(
                "❌ Could not find token with that policy ID or invalid metadata.",
                ephemeral=True
            )
            return
        try:
            threshold = float(self.threshold.value or 1000.0)
        except ValueError:
            await interaction.response.send_message(
                "❌ Invalid threshold value. Please enter a number.",
                ephemeral=True
            )
            return
        track_transfers = self.track_transfers.value.lower() != 'no'
        
        tracker = TokenTrackerClass(
            self.policy_id.value, interaction.channel_id, self.token_name.value,
            self.image_url.value, threshold, track_transfers, token_info['decimals']
        )
        database.add_tracker(
            policy_id=tracker.policy_id, channel_id=tracker.channel_id,
            token_name=tracker.token_name, image_url=tracker.image_url,
            threshold=tracker.threshold, track_transfers=tracker.track_transfers,
            decimals=tracker.decimals
        )
        embed = discord.Embed(
            title="✅ Token Tracking Started",
            description=f"Successfully initialized tracking for {tracker.token_name}",
            color=discord.Color.blue()
        )
        embed.add_field(name="Token", value=f"```{tracker.token_name}```", inline=True)
        embed.add_field(name="Policy ID", value=f"```{tracker.policy_id}```", inline=False)
        embed.add_field(
            name="Configuration",
            value=f"**Threshold:** {tracker.threshold:,.2f} {tracker.token_name}\n"
                  f"**Transfer Notifications:** {'Enabled' if track_transfers else 'Disabled'}",
            inline=False
        )
        if tracker.image_url:
            embed.set_thumbnail(url=tracker.image_url)
        view = TokenControls(tracker.policy_id)
        await interaction.response.send_message(embed=embed, view=view)
        active_trackers[f"{tracker.policy_id}:{tracker.channel_id}"] = tracker
        logger.info(f"Started tracking token: {tracker.token_name} ({tracker.policy_id})")

@bot.tree.command(name="start", description="Start tracking token purchases and transfers")
async def start(interaction: discord.Interaction):
    """Command to initiate token tracking setup"""
    modal = TokenSetupModal()
    await interaction.response.send_modal(modal)

@bot.tree.command(name="help", description="Show help information about the bot's commands")
async def help_command(interaction: discord.Interaction):
    """Display help information for bot commands"""
    embed = discord.Embed(
        title="Help",
        description="Track Cardano token transactions with real-time notifications!",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="📝 Commands",
        value=(
            "**`/start`**\nStart tracking a token's transactions\n\n"
            "**`/status`**\nView tracked tokens and settings\n\n"
            "**`/help`**\nShow this help message\n\n"
            "**`/stop`**\nStop tracking all tokens in this channel"
        ),
        inline=False
    )
    embed.add_field(
        name="🔍 Monitoring Features",
        value="• DEX Trade Detection (Buy/Sell)\n• Wallet Transfer Tracking\n• Real-time Notifications\n• Customizable Thresholds",
        inline=True
    )
    embed.add_field(
        name="🔔 Notifications Include",
        value="• Trade/Transfer Amount\n• Wallet Addresses\n• Transaction Hash\n• Token Image (if provided)",
        inline=True
    )
    embed.set_footer(text="Need more help? Contact the bot administrator.")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Show status of tracked tokens in this channel")
@app_commands.default_permissions(administrator=True)
async def status_command(interaction: discord.Interaction):
    """Show current tracking status for tokens in this channel"""
    trackers = [t for t in active_trackers.values() if t.channel_id == interaction.channel_id]
    if not trackers:
        embed = discord.Embed(
            title="❌ No Active Trackers",
            description="No tokens are currently being tracked in this channel.",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    for tracker in trackers:
        embed = discord.Embed(
            title=f"Tracking {tracker.token_name}",
            description="Current tracking configuration:",
            color=discord.Color.blue()
        )
        embed.add_field(name="Token", value=f"```{tracker.token_name}```", inline=True)
        embed.add_field(name="Policy ID", value=f"```{tracker.policy_id}```", inline=False)
        embed.add_field(
            name="Settings",
            value=f"**Threshold:** {tracker.threshold:,.2f} {tracker.token_name}\n"
                  f"**Transfer Notifications:** {'Enabled' if tracker.track_transfers else 'Disabled'}",
            inline=False
        )
        embed.add_field(
            name="Statistics",
            value=f"**Trade Notifications:** {tracker.trade_notifications}\n"
                  f"**Transfer Notifications:** {tracker.transfer_notifications}",
            inline=False
        )
        if tracker.image_url:
            embed.set_thumbnail(url=tracker.image_url)
        view = TokenControls(tracker.policy_id)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

@bot.tree.command(name="stop", description="Stop tracking all tokens in this channel")
@app_commands.default_permissions(administrator=True)
async def stop(interaction: discord.Interaction):
    """Stop tracking all tokens in the current channel with confirmation"""
    trackers = [t for t in active_trackers.values() if t.channel_id == interaction.channel_id]
    if not trackers:
        await interaction.response.send_message("No tokens are being tracked.", ephemeral=True)
        return
    embed = discord.Embed(
        title="⚠️ Stop Token Tracking",
        description="Are you sure you want to stop tracking all tokens in this channel?",
        color=discord.Color.yellow()
    )
    tokens_list = "\n".join([f"• {t.token_name} (`{t.policy_id[:8]}...`)" for t in trackers])
    embed.add_field(name="Tokens to remove:", value=tokens_list, inline=False)

    class ConfirmView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=60)

        @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
        async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
            try:
                database.remove_all_trackers_for_channel(interaction.channel_id)
                for key in [k for k in active_trackers if k.endswith(f":{interaction.channel_id}")]:
                    del active_trackers[key]
                for child in self.children:
                    child.disabled = True
                embed = discord.Embed(
                    title="✅ Tracking Stopped",
                    description="All tracking stopped in this channel.",
                    color=discord.Color.green()
                )
                await interaction.response.edit_message(embed=embed, view=self)
                logger.info(f"Stopped all tracking for channel {interaction.channel_id}")
            except Exception as e:
                logger.error(f"Error stopping tracking: {str(e)}")
                await interaction.response.send_message("Failed to stop tracking.", ephemeral=True)

        @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            for child in self.children:
                child.disabled = True
            embed = discord.Embed(
                title="❌ Cancelled",
                description="Token tracking will continue.",
                color=discord.Color.red()
            )
            await interaction.response.edit_message(embed=embed, view=self)

    await interaction.response.send_message(embed=embed, view=ConfirmView())

def run_webhook_server():
    """Run the FastAPI webhook server in a separate thread"""
    port = int(os.getenv('PORT', 8000))
    logger.info(f"Starting webhook server on port {port}")
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()

async def start_bot():
    """Start both the Discord bot and webhook server"""
    # Start webhook server in a separate thread
    server_thread = threading.Thread(target=run_webhook_server, daemon=True)
    server_thread.start()
    
    # Run Discord bot in the main thread
    await bot.start(os.getenv('DISCORD_TOKEN'))

if __name__ == "__main__":
    # Run both services using asyncio
    asyncio.run(start_bot())