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
import queue
from queue import Queue
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables from .env file
load_dotenv()

# Configure logging for both console and Heroku compatibility
def setup_logging():
    """Configure logging with separate handlers for info and errors"""
    log_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setFormatter(log_format)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setFormatter(log_format)
    error_handler.setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.info("Bot logging initialized")
    return logger

logger = setup_logging()

# SQLAlchemy Database Setup
Base = declarative_base()

class TokenTracker(Base):
    """Model for token tracking configuration"""
    __tablename__ = 'trackers'
    
    policy_id = Column(String, primary_key=True)
    channel_id = Column(BigInteger, primary_key=True)
    token_name = Column(String)
    image_url = Column(String)
    threshold = Column(Float, default=1000.0, nullable=False)
    track_transfers = Column(Boolean, default=True, nullable=False)
    last_block = Column(BigInteger, default=0, nullable=False)
    trade_notifications = Column(Integer, default=0, nullable=False)
    transfer_notifications = Column(Integer, default=0, nullable=False)
    token_info = Column(JSON)
    
    def to_dict(self):
        """Convert tracker to dictionary"""
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
    """Database connection and operations handler"""
    def __init__(self, database_url=None):
        """Initialize database connection"""
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("No database URL provided")
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def get_trackers(self):
        """Get all token trackers from the database"""
        session = self.Session()
        try:
            trackers = session.query(TokenTracker).all()
            for tracker in trackers:
                tracker.channel_id = int(tracker.channel_id)
            return trackers
        except SQLAlchemyError as e:
            logger.error(f"Database error while retrieving trackers: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
            
    def add_tracker(self, policy_id, token_name, channel_id, image_url=None, threshold=1000.0, token_info=None, track_transfers=True):
        """Add a new token tracker or update if exists"""
        session = self.Session()
        try:
            tracker = session.query(TokenTracker).filter_by(
                policy_id=policy_id,
                channel_id=channel_id
            ).first()
            if tracker:
                tracker.token_name = token_name
                tracker.image_url = image_url
                tracker.threshold = threshold
                tracker.token_info = token_info
                tracker.track_transfers = track_transfers
                logger.info(f"Updated existing tracker for {token_name}")
            else:
                tracker = TokenTracker(
                    policy_id=policy_id,
                    token_name=token_name,
                    channel_id=channel_id,
                    image_url=image_url,
                    threshold=threshold,
                    token_info=token_info,
                    track_transfers=track_transfers
                )
                session.add(tracker)
                logger.info(f"Created new tracker for {token_name}")
            session.commit()
            return tracker
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error adding/updating tracker: {str(e)}", exc_info=True)
            return None
        finally:
            session.close()
            
    def delete_token_tracker(self, policy_id, channel_id):
        """Remove a token tracker"""
        session = self.Session()
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
            logger.error(f"Database error removing tracker: {str(e)}", exc_info=True)
            session.rollback()
            return False
        finally:
            session.close()

    def remove_all_trackers_for_channel(self, channel_id):
        """Remove all token trackers for a specific channel"""
        session = self.Session()
        try:
            trackers = session.query(TokenTracker).filter_by(channel_id=channel_id).all()
            for tracker in trackers:
                session.delete(tracker)
            session.commit()
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database error removing trackers: {str(e)}", exc_info=True)
            session.rollback()
            return False
        finally:
            session.close()

    def save_token_tracker(self, tracker_data):
        """Save a token tracker to the database"""
        session = self.Session()
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
            return tracker
        except SQLAlchemyError as e:
            logger.error(f"Database error saving tracker: {str(e)}", exc_info=True)
            session.rollback()
            return None
        finally:
            session.close()

    def increment_notification_count(self, policy_id, channel_id, notification_type):
        """Increment the notification count for a token tracker"""
        session = self.Session()
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
                return True
            return False
        except SQLAlchemyError as e:
            logger.error(f"Database error incrementing count: {str(e)}", exc_info=True)
            session.rollback()
            return False
        finally:
            session.close()

# Create database instance
database = Database()


# Initialize FastAPI app for webhook handling
app = FastAPI()
notification_queue = Queue()

# Add CORS middleware for webhook endpoint
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
intents.guilds = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Initialize Blockfrost API for Cardano blockchain data
api = BlockFrostApi(
    project_id=os.getenv('BLOCKFROST_API_KEY'),
    base_url=ApiUrls.mainnet.value
)

# Webhook secret and tolerance for signature verification
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
WEBHOOK_TOLERANCE_SECONDS = 600

# Dictionary to store active token trackers
active_trackers = {}

# TokenTracker class to manage tracking configuration
class TokenTrackerClass:
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
        self.last_block = last_block if last_block is not None else 0
        self.trade_notifications = trade_notifications
        self.transfer_notifications = transfer_notifications
        self.token_info = token_info or get_token_info(policy_id)
        if self.token_info:
            logger.info(f"Token {token_name} has {self.token_info.get('decimals', 0)} decimals")

    def __str__(self):
        return f"TokenTracker(policy_id={self.policy_id}, token_name={self.token_name}, channel_id={self.channel_id})"

    def increment_trade_notifications(self):
        """Increment trade notification counter"""
        self.trade_notifications += 1
        database.increment_notification_count(self.policy_id, self.channel_id, 'trade')

    def increment_transfer_notifications(self):
        """Increment transfer notification counter"""
        self.transfer_notifications += 1
        database.increment_notification_count(self.policy_id, self.channel_id, 'transfer')

def verify_webhook_signature(payload: bytes, header: str, current_time: int) -> bool:
    """Verify the Blockfrost webhook signature"""
    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET not set, skipping signature verification")
        return True
    try:
        pairs = dict(pair.split('=') for pair in header.split(','))
        if 't' not in pairs or 'v1' not in pairs:
            logger.error("Missing timestamp or signature in header")
            return False
        timestamp = pairs['t']
        signature = pairs['v1']
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
        logger.error(f"Error verifying webhook signature: {str(e)}", exc_info=True)
        return False

@app.post("/webhook/transaction")
async def transaction_webhook(request: Request):
    """Handle incoming transaction webhooks from Blockfrost"""
    try:
        logger.info("=== Webhook Request Received ===")
        signature = request.headers.get('Blockfrost-Signature')
        if not signature:
            logger.error("❌ Missing Blockfrost-Signature header")
            raise HTTPException(status_code=400, detail="Missing signature header")
        
        payload = await request.body()
        logger.info(f"📦 Raw payload size: {len(payload)} bytes")
        
        current_time = int(time.time())
        if not verify_webhook_signature(payload, signature, current_time):
            logger.error("❌ Invalid webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        data = await request.json()
        tx_count = len(data.get('payload', []))
        logger.info(f"📥 Webhook contains {tx_count} transaction(s)")
        
        if not isinstance(data, dict) or 'type' not in data or data['type'] != 'transaction':
            logger.error(f"❌ Invalid webhook data format: {data.get('type', 'unknown type')}")
            return {"status": "ignored"}
        
        transactions = data.get('payload', [])
        if not transactions:
            logger.info("ℹ️ Webhook contains no transactions")
            return {"status": "no transactions"}
        
        for tx_data in transactions:
            tx = tx_data.get('tx', {})
            tx_hash = tx.get('hash', 'unknown')
            logger.info(f"🔍 Processing transaction: {tx_hash}")
            
            inputs = tx_data.get('inputs', [])
            outputs = tx_data.get('outputs', [])
            if not tx or not inputs or not outputs:
                logger.warning(f"⚠️ Skipping transaction {tx_hash} - Missing required data")
                continue
            
            trackers = database.get_trackers()
            logger.info(f"📋 Checking transaction against {len(trackers)} tracked tokens")
            
            for tracker in trackers:
                logger.info(f"🔍 Checking tracker: {tracker.token_name} ({tracker.policy_id})")
                tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
                if tracker_key not in active_trackers:
                    logger.info(f"📝 Creating new tracker instance for {tracker.token_name}")
                    active_trackers[tracker_key] = TokenTrackerClass(
                        policy_id=tracker.policy_id,
                        channel_id=int(tracker.channel_id),
                        token_name=tracker.token_name,
                        image_url=tracker.image_url,
                        threshold=tracker.threshold,
                        track_transfers=tracker.track_transfers,
                        last_block=tracker.last_block,
                        trade_notifications=tracker.trade_notifications,
                        transfer_notifications=tracker.transfer_notifications,
                        token_info=tracker.token_info
                    )
                active_tracker = active_trackers[tracker_key]
                
                logger.info(f"🔍 Analyzing transaction for {active_tracker.token_name}")
                analysis_data = {
                    'inputs': inputs,
                    'outputs': outputs,
                    'tx': tx
                }
                tx_type, ada_amount, token_amount, details = analyze_transaction_improved(analysis_data, active_tracker.policy_id)
                details['hash'] = tx_hash
                logger.info(f"📊 Analysis results: type={tx_type}, ADA={ada_amount:.2f}, Tokens={token_amount:.2f}")
                
                if tx_type:  # Only queue if we found a relevant transaction
                    notification_data = {
                        'tx_details': analysis_data,
                        'policy_id': active_tracker.policy_id,
                        'ada_amount': ada_amount,
                        'token_amount': token_amount,
                        'analysis_details': {
                            'type': tx_type,
                            'hash': tx_hash,
                            **details
                        },
                        'tracker': active_tracker
                    }
                    notification_queue.put(notification_data)
                    logger.info(f"✅ Queued notification for {active_tracker.token_name}")
                else:
                    logger.info(f"ℹ️ No relevant transaction found for {active_tracker.token_name}")
        
        logger.info("=== Webhook Processing Complete ===")
        return {"status": "success", "message": "Notification queued"}
    except Exception as e:
        logger.error(f"❌ Error in webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@tasks.loop(seconds=1)
async def process_notification_queue():
    """Process any pending notifications in the queue"""
    try:
        while not notification_queue.empty():
            logger.info("=== Processing Notification from Queue ===")
            data = notification_queue.get_nowait()
            try:
                tx_details = data['tx_details']
                tracker = data['tracker']
                ada_amount = data['ada_amount']
                token_amount = data['token_amount']
                analysis_details = data['analysis_details']
                
                logger.info(f"📦 Queue Data:")
                logger.info(f"- Token: {tracker.token_name}")
                logger.info(f"- Type: {analysis_details['type']}")
                logger.info(f"- Amount: {token_amount:,.2f}")
                logger.info(f"- ADA: {ada_amount:,.2f}")
                
                await send_transaction_notification(
                    tracker=tracker,
                    tx_type=analysis_details['type'],
                    ada_amount=ada_amount,
                    token_amount=token_amount,
                    details=analysis_details
                )
                logger.info("✅ Successfully processed notification")
                notification_queue.task_done()
            except Exception as e:
                logger.error(f"❌ Error processing notification: {str(e)}", exc_info=True)
                notification_queue.task_done()
    except Exception as e:
        logger.error(f"❌ Error in notification queue loop: {str(e)}", exc_info=True)

@bot.event
async def on_ready():
    """Called when the bot is ready"""
    try:
        if not process_notification_queue.is_running():
            process_notification_queue.start()
            logger.info("Notification queue processor started")
        
        trackers = database.get_trackers()
        logger.info(f"Loading {len(trackers)} trackers from database")
        
        for tracker in trackers:
            try:
                tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
                active_trackers[tracker_key] = TokenTrackerClass(
                    policy_id=tracker.policy_id,
                    channel_id=int(tracker.channel_id),
                    token_name=tracker.token_name,
                    image_url=tracker.image_url,
                    threshold=tracker.threshold,
                    track_transfers=tracker.track_transfers,
                    last_block=tracker.last_block,
                    trade_notifications=tracker.trade_notifications,
                    transfer_notifications=tracker.transfer_notifications,
                    token_info=tracker.token_info
                )
                logger.info(f"Loaded tracker for {tracker.token_name} in channel {tracker.channel_id}")
            except Exception as e:
                logger.error(f"Error loading tracker {tracker.policy_id}: {str(e)}", exc_info=True)
        
        await bot.tree.sync()
        logger.info(f"Bot is ready! Loaded {len(active_trackers)} trackers")
        
        for guild in bot.guilds:
            logger.info(f"Connected to guild: {guild.name} (ID: {guild.id})")
            logger.info(f"Available channels: {[f'{c.name} (ID: {c.id})' for c in guild.channels]}")
    except Exception as e:
        logger.error(f"Error in on_ready: {str(e)}", exc_info=True)

def get_token_info(policy_id: str):
    """Get token information including metadata and decimals"""
    try:
        assets = api.assets_policy(policy_id)
        if isinstance(assets, Exception):
            raise assets
        if not assets:
            return None
        
        asset = assets[0]
        metadata = api.asset(asset.asset)
        if isinstance(metadata, Exception):
            raise metadata
        
        def namespace_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return [namespace_to_dict(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: namespace_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        metadata_dict = namespace_to_dict(metadata)
        onchain_metadata = metadata_dict.get('onchain_metadata', {})
        
        decimals = None
        if onchain_metadata and isinstance(onchain_metadata, dict):
            decimals = onchain_metadata.get('decimals')
        if decimals is None and 'metadata' in metadata_dict:
            decimals = metadata_dict['metadata'].get('decimals')
        if decimals is None:
            decimals = 0
            logger.info(f"No decimal info for {asset.asset}, defaulting to 0")
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
    formatted = amount / (10 ** decimals)
    if decimals <= 2:
        return f"{formatted:,.{decimals}f}"
    elif formatted >= 1000:
        return f"{formatted:,.2f}"
    else:
        return f"{formatted:,.{min(decimals, 6)}f}"

def get_transaction_details(api: BlockFrostApi, tx_hash: str):
    """Fetch detailed transaction information"""
    try:
        logger.info(f"Fetching transaction details for tx_hash: {tx_hash}")
        tx = api.transaction(tx_hash)
        if isinstance(tx, Exception):
            raise tx
        logger.debug(f"Transaction details retrieved: {tx_hash}")
        
        utxos = api.transaction_utxos(tx_hash)
        if isinstance(utxos, Exception):
            raise utxos
        logger.debug(f"Transaction UTXOs retrieved: {tx_hash}")
        
        metadata = api.transaction_metadata(tx_hash)
        if isinstance(metadata, Exception):
            raise metadata
        logger.debug(f"Transaction metadata retrieved: {tx_hash}")
        
        return tx, utxos, metadata
    except ApiError as e:
        logger.error(f"Blockfrost API error: {str(e)}")
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error in get_transaction_details: {str(e)}", exc_info=True)
        return None, None, None

def analyze_transaction_improved(tx_details, policy_id):
    """Enhanced transaction analysis"""
    try:
        inputs = tx_details.get('inputs', [])
        outputs = tx_details.get('outputs', [])
        if not inputs and not outputs:
            logger.warning("No inputs/outputs found in transaction")
            return 'unknown', 0, 0, {}

        ada_in, ada_out, token_in, token_out = 0, 0, 0, 0
        token_info = get_token_info(policy_id)
        decimals = token_info.get('decimals', 0) if token_info else 0
        full_asset_name = f"{policy_id}{token_info['name']}" if token_info and 'name' in token_info else None
        
        for inp in inputs:
            for amount in inp.get('amount', []):
                unit = amount.get('unit', '')
                logger.debug(f"Checking input unit: {unit}")
                if full_asset_name and unit == full_asset_name or unit.startswith(policy_id):
                    raw_amount = int(amount['quantity'])
                    token_in += raw_amount
                    logger.info(f"Found {raw_amount} tokens in input with unit {unit}")
                elif unit == 'lovelace':
                    ada_in += int(amount['quantity'])

        for out in outputs:
            for amount in out.get('amount', []):
                unit = amount.get('unit', '')
                logger.debug(f"Checking output unit: {unit}")
                if full_asset_name and unit == full_asset_name or unit.startswith(policy_id):
                    raw_amount = int(amount['quantity'])
                    token_out += raw_amount
                    logger.info(f"Found {raw_amount} tokens in output with unit {unit}")
                elif unit == 'lovelace':
                    ada_out += int(amount['quantity'])

        ada_in = ada_in / 1_000_000
        ada_out = ada_out / 1_000_000
        ada_delta = abs(ada_out - ada_in)

        ada_amount = abs(ada_out - ada_in)
        if token_in > 0 and token_out > 0:
            max_output = 0
            for out in outputs:
                for amount in out.get('amount', []):
                    unit = amount.get('unit', '')
                    if full_asset_name and unit == full_asset_name or unit.startswith(policy_id):
                        output_amount = int(amount['quantity'])
                        max_output = max(max_output, output_amount)
            raw_token_amount = max_output
            logger.info(f"Wallet transfer - using largest output amount: {raw_token_amount}")
        else:
            raw_token_amount = abs(token_out - token_in)
            logger.info(f"Buy/Sell - using difference: {raw_token_amount}")
        
        token_amount = raw_token_amount / (10 ** decimals) if decimals > 0 else raw_token_amount
        logger.info(f"Raw token amount: {raw_token_amount}, Decimals: {decimals}, Converted: {token_amount}")

        logger.info(f"Token input: {token_in}, Token output: {token_out}")
        logger.info(f"ADA input: {ada_in}, ADA output: {ada_out}")

        details = {
            'ada_in': ada_in,
            'ada_out': ada_out,
            'token_in': token_in,
            'token_out': token_out,
            'raw_token_amount': raw_token_amount,
            'decimals': decimals,
            'full_asset_name': full_asset_name
        }
        
        if ada_delta > 1.0:  # More than 1 ADA change indicates a trade
            if token_out > token_in:
                return 'buy', ada_delta, token_amount, details
            else:
                return 'sell', ada_delta, token_amount, details
        elif token_in > 0 and token_out > 0:
            return 'wallet_transfer', ada_delta, token_amount, details
    except Exception as e:
        logger.error(f"Error analyzing transaction: {str(e)}", exc_info=True)
        return 'unknown', 0, 0, {}
            

async def create_trade_embed(tx_details, policy_id, ada_amount, token_amount, tracker, analysis_details):
    """Creates a detailed embed for DEX trades"""
    try:
        trade_type = analysis_details.get("type", "unknown")
        title_emoji = "💰" if trade_type == "buy" else "💱"
        action_word = "Purchase" if trade_type == "buy" else "Sale"
        
        embed = discord.Embed(
            title=f"{title_emoji} Token {action_word} Detected",
            description=f"Transaction Hash: [`{tx_details.get('hash', '')[:8]}...{tx_details.get('hash', '')[-8:]}`](https://pool.pm/tx/{tx_details.get('hash', '')})",
            color=discord.Color.green() if trade_type == "buy" else discord.Color.blue()
        )

        overview = (
            "```\n"
            f"Type     : DEX {action_word}\n"
            f"Status   : Confirmed\n"
            f"Addresses: {len(tx_details.get('inputs', [])) + len(tx_details.get('outputs', []))}\n"
            "```"
        )
        embed.add_field(name="📝 Overview", value=overview, inline=True)

        if trade_type == "buy":
            trade_info = (
                "```\n"
                f"ADA Spent  : {ada_amount:,.2f}\n"
                f"Tokens Recv: {format_token_amount(int(token_amount * 10**tracker.token_info.get('decimals', 0)), tracker.token_info.get('decimals', 0))}\n"
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
        embed.add_field(name="💰 Trade Details", value=trade_info, inline=True)

        if tracker.token_info:
            token_name = tracker.token_info.get('name', 'Unknown Token')
            embed.set_author(name=token_name)
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)

        input_addresses = [inp.get('address', '') for inp in tx_details.get('inputs', [])]
        output_addresses = [out.get('address', '') for out in tx_details.get('outputs', [])]
        address_layout = []
        in_addrs = ["📥 Input Addresses:"] + [f"{addr[:8]}..." for addr in input_addresses[:3]]
        if len(input_addresses) > 3:
            in_addrs.append(f"...and {len(input_addresses) - 3} more")
        out_addrs = ["📤 Output Addresses:"] + [f"{addr[:8]}..." for addr in output_addresses[:3]]
        if len(output_addresses) > 3:
            out_addrs.append(f"...and {len(output_addresses) - 3} more")
        
        max_lines = max(len(in_addrs), len(out_addrs))
        for i in range(max_lines):
            left = in_addrs[i] if i < len(in_addrs) else ""
            right = out_addrs[i] if i < len(out_addrs) else ""
            address_layout.append(f"{left:<40} {right}")
        
        if address_layout:
            embed.add_field(name="🔍 Addresses", value="\n".join(address_layout), inline=False)

        embed.add_field(name="🔑 Policy ID", value=f"```{policy_id}```", inline=False)
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
        from_address = tx_details.get('inputs', [{}])[0].get('address', '')
        to_address = tx_details.get('outputs', [{}])[0].get('address', '')
        
        transfer_details = (
            f"**From:** ```{from_address[:20]}...{from_address[-8:]}```\n"
            f"**To:** ```{to_address[:20]}...{to_address[-8:]}```\n"
            f"**Amount:** ```{format_token_amount(int(token_amount * 10**tracker.token_info.get('decimals', 0)), tracker.token_info.get('decimals', 0))} Tokens```"
        )
        embed.add_field(name="🔄 Transfer Details", value=transfer_details, inline=False)
        
        wallet_links = (
            f"[View Sender Wallet](https://cardanoscan.io/address/{from_address})\n"
            f"[View Receiver Wallet](https://cardanoscan.io/address/{to_address})"
        )
        embed.add_field(name="👤 Wallet Profiles", value=wallet_links, inline=False)
        
        embed.add_field(
            name="🔍 Transaction Details",
            value=f"[View on CardanoScan](https://cardanoscan.io/transaction/{tx_details.get('hash', '')})",
            inline=False
        )
        
        embed.set_thumbnail(url=tracker.image_url)
        embed.timestamp = discord.utils.utcnow()
        embed.set_footer(text=f"Transfer detected at • Block #{tx_details.get('block_height', '')}")
        
        return embed
    except Exception as e:
        logger.error(f"Error creating transfer embed: {str(e)}", exc_info=True)
        return None

def shorten_address(address):
    """Shortens a Cardano address for display"""
    if not address:
        return "Unknown"
    return address[:8] + "..." + address[-4:] if len(address) > 12 else address

async def send_transaction_notification(tracker, tx_type, ada_amount, token_amount, details):
    """Send a notification about a transaction to the appropriate Discord channel"""
    try:
        logger.info(f"=== Starting Notification Process ===")
        logger.info(f"Token: {tracker.token_name} ({tracker.policy_id})")
        logger.info(f"Type: {tx_type}, Amount: {token_amount:,.2f}, ADA: {ada_amount:,.2f}")
        logger.info(f"Channel ID: {tracker.channel_id}")
        
        if tx_type == 'wallet_transfer' and not tracker.track_transfers:
            logger.info("❌ Transfer tracking disabled, skipping notification")
            return

        decimals = tracker.token_info.get('decimals', 0) if tracker.token_info else 0
        human_readable_amount = token_amount
        logger.info(f"Threshold Check - Amount: {human_readable_amount:,.{decimals}f}, Required: {tracker.threshold:,.{decimals}f}")
        if human_readable_amount < tracker.threshold:
            logger.info(f"❌ Amount below threshold, skipping notification")
            return
        
        logger.info(f"🔍 Fetching channel {tracker.channel_id}")
        channel = bot.get_channel(tracker.channel_id)
        if not channel:
            logger.warning(f"Channel not in cache, attempting to fetch")
            try:
                channel = await bot.fetch_channel(tracker.channel_id)
                logger.info(f"✅ Successfully fetched channel {channel.name} ({channel.id})")
            except discord.NotFound:
                logger.error(f"❌ Channel {tracker.channel_id} not found")
                tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
                if tracker_key in active_trackers:
                    del active_trackers[tracker_key]
                database.delete_token_tracker(tracker.policy_id, tracker.channel_id)
                return
            except discord.Forbidden:
                logger.error(f"❌ No permission to access channel {tracker.channel_id}")
                return
            except Exception as e:
                logger.error(f"❌ Error fetching channel: {str(e)}", exc_info=True)
                return
        
        if not channel:
            logger.error(f"❌ Failed to resolve channel {tracker.channel_id}")
            return
        
        logger.info(f"📝 Creating notification for channel {channel.name}")
        if tx_type in ['buy', 'sell']:
            embed = await create_trade_embed(details, tracker.policy_id, ada_amount, token_amount, tracker, details)
            if embed:
                try:
                    logger.info(f"📨 Sending trade notification")
                    await channel.send(embed=embed)
                    tracker.increment_trade_notifications()
                    logger.info(f"✅ Successfully sent trade notification")
                except discord.Forbidden:
                    logger.error(f"❌ No permission to send messages in channel {channel.name}")
                except Exception as e:
                    logger.error(f"❌ Error sending trade notification: {str(e)}", exc_info=True)
            else:
                logger.error("❌ Failed to create trade embed")
        elif tx_type == 'wallet_transfer' and tracker.track_transfers:
            embed = await create_transfer_embed(details, tracker.policy_id, token_amount, tracker)
            if embed:
                try:
                    logger.info(f"📨 Sending transfer notification")
                    await channel.send(embed=embed)
                    tracker.increment_transfer_notifications()
                    logger.info(f"✅ Successfully sent transfer notification")
                except discord.Forbidden:
                    logger.error(f"❌ No permission to send messages in channel {channel.name}")
                except Exception as e:
                    logger.error(f"❌ Error sending transfer notification: {str(e)}", exc_info=True)
            else:
                logger.error("❌ Failed to create transfer embed")
        logger.info(f"=== Notification Process Complete ===")
    except Exception as e:
        logger.error(f"❌ Error in notification process: {str(e)}", exc_info=True)

class TokenControls(discord.ui.View):
    def __init__(self, policy_id):
        super().__init__(timeout=None)
        self.policy_id = policy_id

    @discord.ui.button(label="Stop Tracking", style=discord.ButtonStyle.danger, emoji="⛔")
    async def stop_tracking(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            tracker_key = f"{self.policy_id}:{interaction.channel_id}"
            if tracker_key in active_trackers:
                database.delete_token_tracker(self.policy_id, interaction.channel_id)
                del active_trackers[tracker_key]
                embed = discord.Embed(
                    title="✅ Token Tracking Stopped",
                    description=f"Stopped tracking token with policy ID: ```{self.policy_id}```",
                    color=discord.Color.green()
                )
                for child in self.children:
                    child.disabled = True
                await interaction.response.edit_message(embed=embed, view=self)
                logger.info(f"Stopped tracking token: {self.policy_id}")
            else:
                await interaction.response.send_message("Not tracking this token anymore.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error stopping token tracking: {str(e)}", exc_info=True)
            await interaction.response.send_message("Failed to stop tracking.", ephemeral=True)

    @discord.ui.button(label="Toggle Transfers", style=discord.ButtonStyle.primary, emoji="🔄")
    async def toggle_transfers(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            tracker_key = f"{self.policy_id}:{interaction.channel_id}"
            tracker = active_trackers.get(tracker_key)
            if not tracker:
                await interaction.response.send_message("Token tracker not found.", ephemeral=True)
                return
            tracker.track_transfers = not tracker.track_transfers
            database.save_token_tracker({
                'policy_id': self.policy_id,
                'token_name': tracker.token_name,
                'image_url': tracker.image_url,
                'threshold': tracker.threshold,
                'channel_id': interaction.channel_id,
                'last_block': tracker.last_block,
                'track_transfers': tracker.track_transfers,
                'token_info': tracker.token_info,
                'trade_notifications': tracker.trade_notifications,
                'transfer_notifications': tracker.transfer_notifications
            })
            embed = discord.Embed(
                title="✅ Token Tracking Active",
                description="Currently tracking the following token:",
                color=discord.Color.blue()
            )
            token_text = (
                f"**Policy ID:** ```{tracker.policy_id}```\n"
                f"**Name:** ```{tracker.token_name}```"
            )
            embed.add_field(name="Token Information", value=token_text, inline=False)
            config_text = (
                f"**Threshold:** ```{tracker.threshold:,.2f} Tokens```\n"
                f"**Transfer Notifications:** ```{'Enabled' if tracker.track_transfers else 'Disabled'}```\n"
            )
            embed.add_field(name="", value=config_text, inline=False)
            stats_text = (
                f"**Trade Notifications:** ```{tracker.trade_notifications}```\n"
                f"**Transfer Notifications:** ```{tracker.transfer_notifications}```\n"
            )
            embed.add_field(name="", value=stats_text, inline=False)
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)
            await interaction.response.edit_message(embed=embed)
            logger.info(f"Toggled transfers for {tracker.token_name} to {tracker.track_transfers}")
        except Exception as e:
            logger.error(f"Error toggling transfers: {str(e)}", exc_info=True)
            await interaction.response.send_message("Failed to toggle transfers.", ephemeral=True)

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
            token_info = get_token_info(self.policy_id.value)
            if not token_info:
                await interaction.response.send_message(
                    "❌ Could not find token with that policy ID.",
                    ephemeral=True
                )
                return
            try:
                threshold = float(self.threshold.value) if self.threshold.value else 1000.0
            except ValueError:
                await interaction.response.send_message(
                    "❌ Invalid threshold value.",
                    ephemeral=True
                )
                return
            track_transfers = self.track_transfers.value.lower() != 'no'
            tracker = TokenTrackerClass(
                policy_id=self.policy_id.value,
                channel_id=interaction.channel_id,
                token_name=self.token_name.value,
                image_url=self.image_url.value if self.image_url.value else None,
                threshold=threshold,
                track_transfers=track_transfers,
                token_info=token_info
            )
            database.add_tracker(
                policy_id=tracker.policy_id,
                channel_id=tracker.channel_id,
                token_name=tracker.token_name,
                image_url=tracker.image_url,
                threshold=tracker.threshold,
                track_transfers=tracker.track_transfers,
                token_info=tracker.token_info
            )
            embed = discord.Embed(
                title="Token Tracking Started",
                description="Successfully initialized tracking for:",
                color=discord.Color.blue()
            )
            embed.add_field(name="Token", value=f"```{tracker.token_name}```", inline=True)
            embed.add_field(name="Policy ID", value=f"```{tracker.policy_id}```", inline=False)
            config_text = (
                f"**Threshold:** ```{threshold:,.2f} Tokens```\n"
                f"**Transfer Notifications:** ```{'Enabled' if track_transfers else 'Disabled'}```\n"
            )
            embed.add_field(name="", value=config_text, inline=False)
            stats_text = (
                f"**Trade Notifications:** ```0```\n"
                f"**Transfer Notifications:** ```0```\n"
            )
            embed.add_field(name="", value=stats_text, inline=False)
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)
            view = TokenControls(tracker.policy_id)
            await interaction.response.send_message(embed=embed, view=view)
            tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
            active_trackers[tracker_key] = tracker
            logger.info(f"Started tracking token: {tracker.token_name} ({tracker.policy_id})")
        except Exception as e:
            logger.error(f"Error in token setup: {str(e)}", exc_info=True)
            error_embed = discord.Embed(
                title="❌ Error",
                description="Failed to start tracking.",
                color=discord.Color.red()
            )
            await interaction.response.send_message(embed=error_embed)

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
    embed.add_field(
        name="📝 Commands",
        value=(
            "**`/start`**\nStart tracking a token's transactions\n\n"
            "**`/status`**\nView tracked tokens and settings\n\n"
            "**`/help`**\nShow this help message"
        ),
        inline=False
    )
    embed.add_field(
        name="🔍 Monitoring Features",
        value="• DEX Trade Detection\n• Wallet Transfer Tracking\n• Real-time Notifications\n• Customizable Thresholds",
        inline=True
    )
    embed.add_field(
        name="🔔 Notifications Include",
        value="• Trade Amount\n• Wallet Addresses\n• Transaction Hash",
        inline=True
    )
    embed.set_footer(text="Need more help? Contact the bot administrator.")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Show status of tracked tokens in this channel")
@app_commands.default_permissions(administrator=True)
async def status_command(interaction: discord.Interaction):
    """Show status of tracked tokens in this channel"""
    try:
        channel_trackers = [t for t in active_trackers.values() if t.channel_id == interaction.channel_id]
        if not channel_trackers:
            embed = discord.Embed(
                title="❌ No Active Trackers",
                description="No tokens are currently being tracked in this channel.",
                color=discord.Color.red()
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        for tracker in channel_trackers:
            embed = discord.Embed(
                title="Token Tracking Active",
                description="Currently tracking the following token:",
                color=discord.Color.blue()
            )
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
            config_text = (
                f"**Threshold:** ```{tracker.threshold:,.2f} Tokens```\n"
                f"**Transfer Notifications:** ```{'Enabled' if tracker.track_transfers else 'Disabled'}```\n"
            )
            embed.add_field(name="", value=config_text, inline=False)
            stats_text = (
                f"**Trade Notifications:** ```{tracker.trade_notifications}```\n"
                f"**Transfer Notifications:** ```{tracker.transfer_notifications}```\n"
            )
            embed.add_field(name="", value=stats_text, inline=False)
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)
            view = TokenControls(tracker.policy_id)
            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
    except Exception as e:
        logger.error(f"Error in status command: {str(e)}", exc_info=True)
        error_embed = discord.Embed(
            title="❌ Error",
            description="Failed to retrieve status.",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=error_embed, ephemeral=True)

@bot.tree.command(name="stop", description="Stop tracking all tokens in this channel")
@app_commands.default_permissions(administrator=True)
async def stop(interaction: discord.Interaction):
    """Stop tracking all tokens in this channel"""
    try:
        channel_trackers = [t for t in active_trackers.values() if t.channel_id == interaction.channel_id]
        if not channel_trackers:
            await interaction.response.send_message("No tokens are being tracked.", ephemeral=True)
            return
        embed = discord.Embed(
            title="⚠️ Stop Token Tracking",
            description="Are you sure you want to stop tracking all tokens in this channel?",
            color=discord.Color.yellow()
        )
        tokens_list = "\n".join([f"• {t.token_name} (`{t.policy_id}`)" for t in channel_trackers])
        embed.add_field(name="Tokens to remove:", value=tokens_list, inline=False)

        class ConfirmView(discord.ui.View):
            def __init__(self):
                super().__init__(timeout=60)

            @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
            async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
                try:
                    database.remove_all_trackers_for_channel(interaction.channel_id)
                    policies_to_remove = [k for k, v in active_trackers.items() if v.channel_id == interaction.channel_id]
                    for policy_id in policies_to_remove:
                        del active_trackers[policy_id]
                    for child in self.children:
                        child.disabled = True
                    embed = discord.Embed(
                        title="✅ Tracking Stopped",
                        description="All tracking stopped in this channel.",
                        color=discord.Color.green()
                    )
                    await interaction.response.edit_message(embed=embed, view=self)
                except Exception as e:
                    logger.error(f"Error stopping tracking: {str(e)}", exc_info=True)
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
    except Exception as e:
        logger.error(f"Error in stop command: {str(e)}", exc_info=True)
        await interaction.response.send_message("Failed to process stop command.", ephemeral=True)

def run_webhook_server():
    """Run the FastAPI webhook server"""
    port = int(os.getenv('PORT', 8000))
    logger.info(f"Starting webhook server on port {port}")
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    return server

async def start_bot():
    """Start both the Discord bot and webhook server"""
    # Start webhook server in a separate thread
    server = run_webhook_server()
    server_thread = threading.Thread(target=server.run)
    server_thread.start()
    
    # Run Discord bot in main thread
    await bot.start(os.getenv('DISCORD_TOKEN'))

if __name__ == "__main__":
    # Run both services using asyncio
    asyncio.run(start_bot())