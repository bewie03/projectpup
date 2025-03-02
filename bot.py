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
import time  # Added missing import
from queue import Queue
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Configure logging with professional standards
logger = logging.getLogger('CardanoTokenTracker')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('cardano_tracker.log', maxBytes=1048576, backupCount=5)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console_handler)

# SQLAlchemy Database Setup
Base = declarative_base()

class TokenTracker(Base):
    """Model for tracking Cardano token configurations with detailed metadata"""
    __tablename__ = 'trackers'
    policy_id = Column(String, primary_key=True)
    channel_id = Column(BigInteger, primary_key=True)
    token_name = Column(String, nullable=False)
    image_url = Column(String, nullable=True)
    threshold = Column(Float, default=1000.0, nullable=False)
    track_transfers = Column(Boolean, default=True, nullable=False)
    last_block = Column(BigInteger, default=0, nullable=False)
    trade_notifications = Column(Integer, default=0, nullable=False)
    transfer_notifications = Column(Integer, default=0, nullable=False)
    token_info = Column(JSON, nullable=True)

    def to_dict(self):
        """Convert tracker instance to a dictionary for serialization"""
        return {
            'policy_id': self.policy_id,
            'channel_id': self.channel_id,
            'token_name': self.token_name,
            'image_url': self.image_url,
            'threshold': self.threshold,
            'track_transfers': self.track_transfers,
            'last_block': self.last_block,
            'trade_notifications': self.trade_notifications,
            'transfer_notifications': self.transfer_notifications,
            'token_info': self.token_info
        }

class Database:
    """Robust database handler for Cardano token tracking with comprehensive error handling"""
    def __init__(self, database_url=None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
        self.engine = create_engine(self.database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

    def get_trackers(self):
        """Retrieve all token trackers, ensuring channel_id is an integer"""
        with self.Session() as session:
            try:
                trackers = session.query(TokenTracker).all()
                for tracker in trackers:
                    tracker.channel_id = int(tracker.channel_id)
                return trackers
            except SQLAlchemyError as e:
                logger.error(f"Failed to retrieve trackers: {str(e)}", exc_info=True)
                return []

    def get_tracker(self, policy_id, channel_id):
        """Fetch a specific token tracker by policy_id and channel_id"""
        with self.Session() as session:
            try:
                tracker = session.query(TokenTracker).filter_by(
                    policy_id=policy_id,
                    channel_id=channel_id
                ).first()
                if tracker:
                    tracker.channel_id = int(tracker.channel_id)
                return tracker
            except SQLAlchemyError as e:
                logger.error(f"Failed to retrieve tracker for {policy_id}/{channel_id}: {str(e)}", exc_info=True)
                return None

    def add_tracker(self, policy_id, token_name, channel_id, image_url=None, threshold=1000.0, token_info=None, track_transfers=True):
        """Add or update a token tracker, ensuring unique policy_id per channel"""
        with self.Session() as session:
            try:
                tracker = session.query(TokenTracker).filter_by(
                    policy_id=policy_id,
                    channel_id=channel_id
                ).first()
                if tracker:
                    tracker.token_name = token_name
                    tracker.image_url = image_url
                    tracker.threshold = threshold
                    tracker.token_info = token_info or tracker.token_info
                    tracker.track_transfers = track_transfers
                    logger.info(f"Updated tracker for {token_name} in channel {channel_id}")
                else:
                    existing_trackers = session.query(TokenTracker).filter_by(policy_id=policy_id).all()
                    last_block = max([t.last_block for t in existing_trackers], default=0)
                    tracker = TokenTracker(
                        policy_id=policy_id,
                        token_name=token_name,
                        channel_id=channel_id,
                        image_url=image_url,
                        threshold=threshold,
                        token_info=token_info,
                        track_transfers=track_transfers,
                        last_block=last_block
                    )
                    session.add(tracker)
                    logger.info(f"Created new tracker for {token_name} in channel {channel_id}")
                session.commit()
                return tracker
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to add/update tracker: {str(e)}", exc_info=True)
                return None

    def delete_token_tracker(self, policy_id, channel_id):
        """Remove a specific token tracker from the database"""
        with self.Session() as session:
            try:
                tracker = session.query(TokenTracker).filter_by(
                    policy_id=policy_id,
                    channel_id=channel_id
                ).first()
                if tracker:
                    session.delete(tracker)
                    session.commit()
                    logger.info(f"Deleted tracker for {policy_id} in channel {channel_id}")
                    return True
                logger.warning(f"No tracker found for {policy_id} in channel {channel_id}")
                return False
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to delete tracker: {str(e)}", exc_info=True)
                return False

    def remove_all_trackers_for_channel(self, channel_id):
        """Remove all trackers associated with a specific Discord channel"""
        with self.Session() as session:
            try:
                affected_rows = session.query(TokenTracker).filter_by(channel_id=channel_id).delete()
                session.commit()
                if affected_rows:
                    logger.info(f"Removed all trackers for channel {channel_id}")
                    return True
                logger.warning(f"No trackers found for channel {channel_id}")
                return False
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to remove trackers for channel {channel_id}: {str(e)}", exc_info=True)
                return False

    def save_token_tracker(self, tracker_data):
        """Save or update a token tracker with the provided data"""
        with self.Session() as session:
            try:
                existing = session.query(TokenTracker).filter_by(
                    policy_id=tracker_data['policy_id'],
                    channel_id=tracker_data['channel_id']
                ).first()
                if existing:
                    for key, value in tracker_data.items():
                        if key == 'last_block' and value is None:
                            value = 0
                        setattr(existing, key, value)
                    tracker = existing
                else:
                    if 'last_block' not in tracker_data or tracker_data['last_block'] is None:
                        tracker_data['last_block'] = 0
                    tracker = TokenTracker(**tracker_data)
                    session.add(tracker)
                session.commit()
                logger.info(f"Saved tracker for {tracker_data['token_name']} in channel {tracker_data['channel_id']}")
                return tracker
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to save tracker: {str(e)}", exc_info=True)
                return None

    def update_last_block(self, policy_id, channel_id, block_height):
        """Update the last processed block height for a tracker"""
        with self.Session() as session:
            try:
                tracker = session.query(TokenTracker).filter_by(
                    policy_id=policy_id,
                    channel_id=channel_id
                ).first()
                if tracker:
                    tracker.last_block = block_height if block_height is not None else 0
                    session.commit()
                    logger.info(f"Updated last block for {policy_id} to {block_height}")
                    return True
                logger.warning(f"No tracker found for {policy_id} in channel {channel_id}")
                return False
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to update last block: {str(e)}", exc_info=True)
                return False

    def increment_notification_count(self, policy_id, channel_id, notification_type):
        """Increment the appropriate notification counter for a tracker"""
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
                    logger.info(f"Incremented {notification_type} notification for {policy_id}")
                    return True
                logger.warning(f"No tracker found for {policy_id} in channel {channel_id}")
                return False
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to increment notification count: {str(e)}", exc_info=True)
                return False

    def get_all_token_trackers(self):
        """Retrieve all token trackers with integer channel_ids"""
        with self.Session() as session:
            try:
                trackers = session.query(TokenTracker).all()
                for tracker in trackers:
                    tracker.channel_id = int(tracker.channel_id)
                return trackers
            except SQLAlchemyError as e:
                logger.error(f"Failed to retrieve all trackers: {str(e)}", exc_info=True)
                return []

    def remove_tracker(self, policy_id, channel_id):
        """Alias for delete_token_tracker with type conversion for channel_id"""
        return self.delete_token_tracker(policy_id, int(channel_id))

database = Database()

# FastAPI setup for webhook handling
app = FastAPI(title="Cardano Token Tracker Webhook", version="1.0.0")
notification_queue = Queue()

# CORS middleware for webhook endpoint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Blockfrost-Signature", "Content-Type"],
)

# Discord bot setup with advanced intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Blockfrost API initialization
api = BlockFrostApi(
    project_id=os.getenv('BLOCKFROST_API_KEY'),
    base_url=ApiUrls.mainnet.value
)

WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', '')
WEBHOOK_TOLERANCE_SECONDS = 600
active_trackers = {}

class TokenTrackerClass:
    """In-memory representation of a token tracker with dynamic state"""
    def __init__(self, policy_id: str, channel_id: int, token_name: str, image_url: str, threshold: float,
                 track_transfers: bool, token_info: dict):
        self.policy_id = policy_id
        self.channel_id = channel_id
        self.token_name = token_name
        self.image_url = image_url
        self.threshold = threshold
        self.track_transfers = track_transfers
        self.token_info = token_info or {}
        self.decimals = self.token_info.get('decimals', 0) if self.token_info and 'decimals' in self.token_info else 0
        self.trade_notifications = 0
        self.transfer_notifications = 0

    def increment_trade_notifications(self):
        """Increment trade notification count and update database"""
        self.trade_notifications += 1
        database.increment_notification_count(self.policy_id, self.channel_id, 'trade')

    def increment_transfer_notifications(self):
        """Increment transfer notification count and update database"""
        self.transfer_notifications += 1
        database.increment_notification_count(self.policy_id, self.channel_id, 'transfer')

def verify_webhook_signature(payload: bytes, header: str, current_time: int) -> bool:
    """Securely verify Blockfrost webhook signatures with robust error handling"""
    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET not configured, skipping signature verification")
        return True
    try:
        pairs = dict(pair.split('=') for pair in header.split(','))
        timestamp = pairs.get('t')
        signature = pairs.get('v1')
        if not timestamp or not signature:
            logger.error("Missing timestamp or signature in webhook header")
            return False
        timestamp_diff = abs(current_time - int(timestamp))
        if timestamp_diff > WEBHOOK_TOLERANCE_SECONDS:
            logger.error(f"Webhook timestamp too old: {timestamp_diff} seconds")
            return False
        signature_payload = f"{timestamp}.{payload.decode('utf-8')}"
        computed = hmac.new(
            WEBHOOK_SECRET.encode('utf-8'),
            signature_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(computed, signature)
    except Exception as e:
        logger.error(f"Signature verification failed: {str(e)}", exc_info=True)
        return False

@app.post("/webhook/transaction")
async def transaction_webhook(request: Request):
    """Handle incoming Cardano transaction webhooks from Blockfrost with precision"""
    try:
        logger.info("Received transaction webhook")
        signature = request.headers.get('Blockfrost-Signature')
        if not signature:
            raise HTTPException(status_code=400, detail="Missing Blockfrost-Signature header")

        payload = await request.body()
        current_time = int(time.time())  # Fixed by adding time import
        if not verify_webhook_signature(payload, signature, current_time):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

        data = await request.json()
        if not isinstance(data, dict) or data.get('type') != 'transaction':
            logger.error(f"Invalid webhook data format: {data.get('type', 'unknown')}")
            return {"status": "ignored"}

        transactions = data.get('payload', [])
        if not transactions:
            logger.info("No transactions in webhook payload")
            return {"status": "processed, no transactions"}

        for tx_data in transactions:
            tx = tx_data.get('tx', {})
            tx_hash = tx.get('hash', 'unknown')
            logger.info(f"Processing transaction {tx_hash}")
            inputs = tx_data.get('inputs', [])
            outputs = tx_data.get('outputs', [])
            if not inputs or not outputs:
                logger.warning(f"Skipping transaction {tx_hash} - missing inputs or outputs")
                continue

            for tracker in database.get_trackers():
                tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
                if tracker_key not in active_trackers:
                    active_trackers[tracker_key] = TokenTrackerClass(
                        tracker.policy_id, int(tracker.channel_id), tracker.token_name,
                        tracker.image_url, tracker.threshold, tracker.track_transfers, tracker.token_info
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
                    logger.info(f"Queued notification for {active_tracker.token_name} ({tx_type})")

        return {"status": "success", "processed": len(transactions)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@tasks.loop(seconds=1)
async def process_notification_queue():
    """Efficiently process queued notifications with error recovery"""
    while not notification_queue.empty():
        try:
            data = notification_queue.get_nowait()
            await send_transaction_notification(**data)
            notification_queue.task_done()
        except Exception as e:
            logger.error(f"Notification queue processing error: {str(e)}", exc_info=True)
            notification_queue.task_done()

@bot.event
async def on_ready():
    """Initialize bot on startup with tracker synchronization and command syncing"""
    if not process_notification_queue.is_running():
        process_notification_queue.start()
    for tracker in database.get_trackers():
        tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
        active_trackers[tracker_key] = TokenTrackerClass(
            tracker.policy_id, int(tracker.channel_id), tracker.token_name,
            tracker.image_url, tracker.threshold, tracker.track_transfers, tracker.token_info
        )
    try:
        await bot.tree.sync()
        logger.info(f"Bot ready and synced with {len(active_trackers)} trackers")
    except Exception as e:
        logger.error(f"Failed to sync bot commands: {str(e)}", exc_info=True)

def get_token_info(policy_id: str) -> dict:
    """Fetch token metadata (decimals, name) from Blockfrost with robust error handling"""
    try:
        assets = api.assets_policy(policy_id)
        if isinstance(assets, Exception) or not assets:
            logger.warning(f"No assets found for policy_id {policy_id}")
            return {}
        asset = assets[0]
        metadata = api.asset(asset.asset)
        if isinstance(metadata, Exception):
            logger.error(f"Failed to fetch metadata for {asset.asset}")
            return {}
        decimals = metadata.onchain_metadata.get('decimals', 0) if metadata.onchain_metadata else 0
        return {
            'decimals': decimals,
            'name': metadata.asset_name or ''
        }
    except ApiError as e:
        logger.error(f"Blockfrost API error for {policy_id}: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error fetching token info: {str(e)}", exc_info=True)
        return {}

def format_token_amount(raw_amount: int, token_info: dict) -> str:
    """Format token amount with precise decimal handling for Cardano tokens"""
    decimals = token_info.get('decimals', 0) if token_info else 0
    if decimals <= 0:
        return f"{raw_amount:,}"
    adjusted_amount = raw_amount / (10 ** decimals)
    return f"{adjusted_amount:,.{min(decimals, 6)}f}"

def analyze_transaction(tx_data, tracker) -> tuple | None:
    """Analyze Cardano transaction to determine type (transfer, buy, sell) and amounts with precise decimal handling"""
    try:
        inputs = tx_data.get('inputs', [])
        outputs = tx_data.get('outputs', [])
        tx = tx_data.get('tx', {})
        tx_hash = tx.get('hash', 'unknown')

        # Calculate ADA and token movements (in raw units)
        ada_in = sum(int(amt['quantity']) for inp in inputs for amt in inp.get('amount', []) if amt['unit'] == 'lovelace')
        ada_out = sum(int(amt['quantity']) for out in outputs for amt in out.get('amount', []) if amt['unit'] == 'lovelace')
        token_sent = sum(int(amt['quantity']) for inp in inputs for amt in inp.get('amount', []) if amt['unit'].startswith(tracker.policy_id))
        token_received = sum(int(amt['quantity']) for out in outputs for amt in out.get('amount', []) if amt['unit'].startswith(tracker.policy_id))

        ada_movement = (ada_out - ada_in) / 1_000_000  # Convert lovelace to ADA
        decimals = tracker.token_info.get('decimals', 0) if tracker.token_info else 0

        # Adjust token amounts for decimals
        token_sent_adjusted = token_sent / (10 ** decimals) if decimals > 0 else token_sent
        token_received_adjusted = token_received / (10 ** decimals) if decimals > 0 else token_received

        # Log for debugging
        logger.debug(f"Tx {tx_hash}: token_sent={token_sent}, token_received={token_received}, ada_movement={ada_movement}")

        # Determine transaction type and amounts
        if abs(ada_movement) > 1.0:  # Likely a DEX trade (significant ADA movement)
            if ada_movement < 0 and token_received > 0:  # ADA spent, tokens received = Buy
                token_amount = token_received_adjusted
                ada_amount = abs(ada_movement)  # ADA spent
                return 'buy', ada_amount, token_amount, {
                    'hash': tx_hash,
                    'inputs': [inp for inp in inputs if inp.get('address')],
                    'outputs': [out for out in outputs if out.get('address')]
                }
            elif ada_movement > 0 and token_sent > 0:  # ADA received, tokens sent = Sell
                token_amount = token_sent_adjusted
                ada_amount = ada_movement  # ADA received
                return 'sell', ada_amount, token_amount, {
                    'hash': tx_hash,
                    'inputs': [inp for inp in inputs if inp.get('address')],
                    'outputs': [out for out in outputs if out.get('address')]
                }
        elif token_sent > 0 or token_received > 0:  # Token transfer between wallets (minimal ADA)
            token_amount = max(token_sent_adjusted, token_received_adjusted)  # Use the larger movement
            return 'transfer', 0, token_amount, {
                'hash': tx_hash,
                'inputs': [inp for inp in inputs if inp.get('address')],
                'outputs': [out for out in outputs if out.get('address')]
                }

        logger.debug(f"No significant transaction detected for {tracker.token_name}")
        return None
    except Exception as e:
        logger.error(f"Transaction analysis failed for {tx_hash}: {str(e)}", exc_info=True)
        return None

async def create_transfer_embed(tracker, token_amount, details):
    """Create a professional embed for token transfer notifications"""
    tx_hash = details['hash']
    inputs = details['inputs']
    outputs = details['outputs']

    # Identify sender and receiver addresses
    sender = next((inp['address'] for inp in inputs if any(amt['unit'].startswith(tracker.policy_id) for amt in inp.get('amount', []))), 'Unknown')
    receiver = next((out['address'] for out in outputs if any(amt['unit'].startswith(tracker.policy_id) for amt in out.get('amount', []))), 'Unknown')

    embed = discord.Embed(
        title=f"↔️ {tracker.token_name} Transfer Detected",
        description=f"[View on CardanoScan](https://cardanoscan.io/transaction/{tx_hash})",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )
    embed.add_field(
        name="Transfer Details",
        value=f"**From:** [{sender[:8]}...](https://cardanoscan.io/address/{sender})\n"
              f"**To:** [{receiver[:8]}...](https://cardanoscan.io/address/{receiver})\n"
              f"**Amount:** {format_token_amount(int(token_amount * (10 ** tracker.token_info.get('decimals', 0))) if tracker.token_info.get('decimals', 0) > 0 else int(token_amount), tracker.token_info)} {tracker.token_name}",
        inline=False
    )
    if tracker.image_url:
        embed.set_thumbnail(url=tracker.image_url)
    embed.set_footer(text=f"Detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    return embed

async def create_trade_embed(tracker, tx_type, ada_amount, token_amount, details):
    """Create a polished embed for DEX trade notifications (buy/sell)"""
    tx_hash = details['hash']
    title = f"💰 {tracker.token_name} {'Purchase' if tx_type == 'buy' else 'Sale'} Detected"
    color = discord.Color.green() if tx_type == 'buy' else discord.Color.red()

    embed = discord.Embed(
        title=title,
        description=f"[View on CardanoScan](https://cardanoscan.io/transaction/{tx_hash})",
        color=color,
        timestamp=datetime.now()
    )
    embed.add_field(
        name="Overview",
        value=f"Type: DEX {'Purchase' if tx_type == 'buy' else 'Sale'}\nStatus: Confirmed",
        inline=True
    )
    embed.add_field(
        name="Trade Details",
        value=f"{'Tokens Bought' if tx_type == 'buy' else 'Tokens Sold'}: {format_token_amount(int(token_amount * (10 ** tracker.token_info.get('decimals', 0))) if tracker.token_info.get('decimals', 0) > 0 else int(token_amount), tracker.token_info)} {tracker.token_name}\n"
              f"{'ADA Spent' if tx_type == 'buy' else 'ADA Received'}: {ada_amount:,.2f} ADA",
        inline=True
    )

    # Extract and format input/output addresses
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
    embed.set_footer(text=f"Detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    return embed

async def send_transaction_notification(tracker, tx_type, ada_amount, token_amount, details):
    """Send transaction notifications to Discord with robust error handling"""
    if tx_type == 'transfer' and not tracker.track_transfers:
        logger.info(f"Skipping transfer notification for {tracker.token_name} - tracking disabled")
        return

    # Use raw token amount for threshold check (adjusted based on decimals)
    decimals = tracker.token_info.get('decimals', 0) if tracker.token_info else 0
    raw_token_amount = int(token_amount * (10 ** decimals)) if decimals > 0 else int(token_amount)

    if raw_token_amount < tracker.threshold:
        logger.info(f"Skipping notification for {tracker.token_name} - raw amount {raw_token_amount} below threshold {tracker.threshold}")
        return

    channel = bot.get_channel(tracker.channel_id)
    if not channel:
        try:
            channel = await bot.fetch_channel(tracker.channel_id)
        except discord.NotFound:
            logger.error(f"Channel {tracker.channel_id} not found, removing tracker")
            database.delete_token_tracker(tracker.policy_id, tracker.channel_id)
            if f"{tracker.policy_id}:{tracker.channel_id}" in active_trackers:
                del active_trackers[f"{tracker.policy_id}:{tracker.channel_id}"]
            return
        except discord.Forbidden:
            logger.error(f"No permission to access channel {tracker.channel_id}")
            return
        except Exception as e:
            logger.error(f"Failed to fetch channel {tracker.channel_id}: {str(e)}", exc_info=True)
            return

    embed = await (create_transfer_embed if tx_type == 'transfer' else create_trade_embed)(
        tracker, token_amount, details if tx_type == 'transfer' else (tx_type, ada_amount, token_amount, details)
    )
    if not embed:
        logger.error(f"Failed to create embed for {tracker.token_name} ({tx_type})")
        return

    try:
        await channel.send(embed=embed)
        if tx_type == 'transfer':
            tracker.increment_transfer_notifications()
        else:
            tracker.increment_trade_notifications()
        logger.info(f"Successfully sent {tx_type} notification for {tracker.token_name}")
    except discord.Forbidden:
        logger.error(f"No permission to send in channel {channel.name} ({channel.id})")
    except Exception as e:
        logger.error(f"Failed to send notification: {str(e)}", exc_info=True)

class TokenControls(discord.ui.View):
    """Interactive controls for managing token tracking with persistent state"""
    def __init__(self, policy_id: str):
        super().__init__(timeout=None)
        self.policy_id = policy_id

    @discord.ui.button(label="Stop Tracking", style=discord.ButtonStyle.danger, emoji="⛔")
    async def stop_tracking(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Stop tracking a specific token in the current channel"""
        if database.delete_token_tracker(self.policy_id, interaction.channel_id):
            tracker_key = f"{self.policy_id}:{interaction.channel_id}"
            if tracker_key in active_trackers:
                del active_trackers[tracker_key]
            embed = discord.Embed(
                title="✅ Tracking Stopped",
                description=f"Stopped tracking token with Policy ID: ```{self.policy_id}```",
                color=discord.Color.green()
            )
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(embed=embed, view=self)
            logger.info(f"Stopped tracking {self.policy_id} in channel {interaction.channel_id}")
        else:
            await interaction.response.send_message("This token is not being tracked here.", ephemeral=True)

    @discord.ui.button(label="Toggle Transfers", style=discord.ButtonStyle.primary, emoji="🔄")
    async def toggle_transfers(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Toggle transfer notification tracking for a token"""
        tracker_key = f"{self.policy_id}:{interaction.channel_id}"
        tracker = active_trackers.get(tracker_key)
        if not tracker:
            await interaction.response.send_message("Token tracker not found.", ephemeral=True)
            return

        tracker.track_transfers = not tracker.track_transfers
        database.add_tracker(
            tracker.policy_id, tracker.token_name, tracker.channel_id,
            tracker.image_url, tracker.threshold, tracker.token_info, tracker.track_transfers
        )
        embed = discord.Embed(
            title="✅ Tracking Updated",
            description=f"Transfer notifications for {tracker.token_name} are now {'enabled' if tracker.track_transfers else 'disabled'}",
            color=discord.Color.blue()
        )
        if tracker.image_url:
            embed.set_thumbnail(url=tracker.image_url)
        await interaction.response.edit_message(embed=embed)
        logger.info(f"Toggled transfers for {tracker.token_name} to {tracker.track_transfers}")

class TokenSetupModal(discord.ui.Modal, title="🪙 Token Setup"):
    """Modal for configuring new token tracking with validation"""
    def __init__(self):
        super().__init__()
        self.policy_id = discord.ui.TextInput(
            label="Policy ID",
            placeholder="Enter the 56-character Cardano policy ID",
            style=discord.TextStyle.short,
            required=True,
            min_length=56,
            max_length=56
        )
        self.token_name = discord.ui.TextInput(
            label="Token Name",
            placeholder="Enter the token name",
            style=discord.TextStyle.short,
            required=True
        )
        self.image_url = discord.ui.TextInput(
            label="Image URL",
            placeholder="Optional token image URL (e.g., PNG/JPEG)",
            style=discord.TextStyle.short,
            required=False
        )
        self.threshold = discord.ui.TextInput(
            label="Minimum Token Amount",
            placeholder="Minimum tokens for notifications (default: 1000)",
            style=discord.TextStyle.short,
            required=False,
            default="1000"
        )
        self.track_transfers = discord.ui.TextInput(
            label="Track Transfers",
            placeholder="'yes' or 'no' to enable/disable transfer notifications (default: yes)",
            style=discord.TextStyle.short,
            required=False,
            default="yes"
        )
        for item in [self.policy_id, self.token_name, self.image_url, self.threshold, self.track_transfers]:
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        """Handle token setup submission with thorough validation"""
        policy_id = self.policy_id.value.strip()
        if len(policy_id) != 56 or not all(c in '0123456789abcdefABCDEF' for c in policy_id):
            await interaction.response.send_message("Invalid Policy ID format (must be 56 hex characters).", ephemeral=True)
            return

        token_info = get_token_info(policy_id)
        if not token_info or 'decimals' not in token_info:
            await interaction.response.send_message("Could not retrieve token metadata or invalid policy ID.", ephemeral=True)
            return

        try:
            threshold = float(self.threshold.value or "1000")
            if threshold < 0:
                raise ValueError
        except ValueError:
            await interaction.response.send_message("Invalid threshold value (must be a positive number).", ephemeral=True)
            return

        track_transfers = self.track_transfers.value.lower() != 'no'

        tracker = TokenTrackerClass(
            policy_id, interaction.channel_id, self.token_name.value.strip(),
            self.image_url.value.strip() if self.image_url.value else None,
            threshold, track_transfers, token_info
        )
        database.add_tracker(
            policy_id, tracker.token_name, tracker.channel_id,
            tracker.image_url, tracker.threshold, tracker.token_info, tracker.track_transfers
        )
        embed = discord.Embed(
            title="✅ Token Tracking Started",
            description=f"Successfully started tracking {tracker.token_name}",
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
        logger.info(f"Started tracking {tracker.token_name} (policy_id: {tracker.policy_id}) in channel {tracker.channel_id}")

@bot.tree.command(name="start", description="Start tracking Cardano token transactions")
async def start(interaction: discord.Interaction):
    """Initiate token tracking setup via modal"""
    await interaction.response.send_modal(TokenSetupModal())

@bot.tree.command(name="help", description="Display help information for the bot")
async def help_command(interaction: discord.Interaction):
    """Provide detailed help about bot functionality"""
    embed = discord.Embed(
        title="Help - Cardano Token Tracker",
        description="Track Cardano token transactions (DEX trades & wallet transfers) with real-time Discord notifications!",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="📜 Commands",
        value=(
            "**`/start`**\nConfigure a new token to track its transactions\n\n"
            "**`/status`**\nView current tracking settings for this channel\n\n"
            "**`/help`**\nShow this help message\n\n"
            "**`/stop`**\nStop tracking all tokens in this channel"
        ),
        inline=False
    )
    embed.add_field(
        name="🔍 Features",
        value="• Real-time DEX Trade Detection (Buy/Sell)\n• Wallet Transfer Tracking\n• Customizable Notification Thresholds\n• Token Image Support",
        inline=True
    )
    embed.add_field(
        name="🔔 Notifications",
        value="• Transaction Type & Amount\n• Wallet Addresses (Input/Output)\n• CardanoScan Links\n• Token Image (if provided)",
        inline=True
    )
    embed.set_footer(text="For support, contact the bot administrator or check our documentation.")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="View tracked tokens in this channel")
@app_commands.default_permissions(manage_channels=True)
async def status_command(interaction: discord.Interaction):
    """Display current tracking status for tokens in this channel"""
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
@app_commands.default_permissions(manage_channels=True)
async def stop(interaction: discord.Interaction):
    """Stop tracking all tokens in the current channel with confirmation"""
    trackers = [t for t in active_trackers.values() if t.channel_id == interaction.channel_id]
    if not trackers:
        await interaction.response.send_message("No tokens are being tracked in this channel.", ephemeral=True)
        return

    embed = discord.Embed(
        title="⚠️ Confirm Stop Tracking",
        description="Are you sure you want to stop tracking all tokens in this channel?",
        color=discord.Color.yellow()
    )
    tokens_list = "\n".join([f"• {t.token_name} (Policy: {t.policy_id[:8]}...)" for t in trackers])
    embed.add_field(name="Tokens to Stop Tracking", value=tokens_list, inline=False)

    class ConfirmView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=300)  # 5-minute timeout for confirmation

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
                    description="All token tracking has been stopped in this channel.",
                    color=discord.Color.green()
                )
                await interaction.response.edit_message(embed=embed, view=self)
                logger.info(f"Stopped all tracking for channel {interaction.channel_id}")
            except Exception as e:
                logger.error(f"Failed to stop tracking: {str(e)}", exc_info=True)
                await interaction.response.send_message("Failed to stop tracking due to an error.", ephemeral=True)

        @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            for child in self.children:
                child.disabled = True
            embed = discord.Embed(
                title="❌ Cancelled",
                description="Token tracking will continue in this channel.",
                color=discord.Color.red()
            )
            await interaction.response.edit_message(embed=embed, view=self)

    await interaction.response.send_message(embed=embed, view=ConfirmView())

def run_webhook_server():
    """Run the FastAPI webhook server with configuration for scalability"""
    port = int(os.getenv('PORT', 8000))
    logger.info(f"Starting webhook server on port {port}")
    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level="info",
        workers=2, timeout_keep_alive=60
    )
    server = uvicorn.Server(config)
    server.run()

async def start_bot():
    """Launch both the Discord bot and webhook server with graceful error handling"""
    server_thread = threading.Thread(target=run_webhook_server, daemon=True)
    server_thread.start()
    try:
        await bot.start(os.getenv('DISCORD_TOKEN'))
    except Exception as e:
        logger.error(f"Bot startup failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    """Main entry point to start the application"""
    asyncio.run(start_bot())