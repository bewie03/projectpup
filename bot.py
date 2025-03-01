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

# Placeholder for database module (implement as needed)
class Database:
    def get_trackers(self):
        return []  # Return list of tracker objects
    def add_tracker(self, **kwargs):
        pass
    def delete_token_tracker(self, policy_id, channel_id):
        pass
    def save_token_tracker(self, tracker_dict):
        pass
    def remove_all_trackers_for_channel(self, channel_id):
        pass

database = Database()  # Replace with your actual database implementation

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setFormatter(formatter)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.WARNING)
    logger.handlers = [info_handler, error_handler]
    logger.info("Bot logging initialized")
    return logger

logger = setup_logging()

# Initialize FastAPI app
app = FastAPI()
notification_queue = Queue()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Initialize Blockfrost API
api = BlockFrostApi(
    project_id=os.getenv('BLOCKFROST_API_KEY'),
    base_url=ApiUrls.mainnet.value
)

# Webhook secret
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
WEBHOOK_TOLERANCE_SECONDS = 600

# Active trackers
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
        self.token_info = token_info or get_token_info(policy_id)

    def increment_trade_notifications(self):
        self.trade_notifications += 1

    def increment_transfer_notifications(self):
        self.transfer_notifications += 1

def verify_webhook_signature(payload: bytes, header: str, current_time: int) -> bool:
    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET not set, skipping verification")
        return True
    try:
        pairs = dict(pair.split('=') for pair in header.split(','))
        timestamp, signature = pairs['t'], pairs['v1']
        if abs(current_time - int(timestamp)) > WEBHOOK_TOLERANCE_SECONDS:
            logger.error(f"Webhook timestamp too old: {abs(current_time - int(timestamp))}s")
            return False
        computed = hmac.new(
            WEBHOOK_SECRET.encode(),
            f"{timestamp}.{payload.decode('utf-8')}".encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(computed, signature)
    except Exception as e:
        logger.error(f"Signature verification error: {str(e)}", exc_info=True)
        return False

@app.post("/webhook/transaction")
async def transaction_webhook(request: Request):
    """Handle Blockfrost transaction webhooks"""
    try:
        logger.info("Received webhook request")
        signature = request.headers.get('Blockfrost-Signature')
        if not signature:
            logger.error("Missing Blockfrost-Signature header")
            raise HTTPException(status_code=400, detail="Missing signature header")

        payload = await request.body()
        current_time = int(time.time())
        if not verify_webhook_signature(payload, signature, current_time):
            logger.error("Invalid webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

        data = await request.json()
        if not isinstance(data, dict) or 'type' not in data or data['type'] != 'transaction':
            logger.error(f"Invalid webhook type: {data.get('type', 'unknown')}")
            return {"status": "ignored"}

        transactions = data.get('payload', [])
        logger.info(f"Webhook contains {len(transactions)} transactions")

        for tx_data in transactions:
            tx = tx_data.get('tx', {})
            tx_hash = tx.get('hash', 'unknown')
            inputs = tx_data.get('inputs', [])
            outputs = tx_data.get('outputs', [])
            if not all([tx, inputs, outputs]):
                logger.warning(f"Skipping tx {tx_hash} - missing data")
                continue

            trackers = database.get_trackers()
            for tracker in trackers:
                tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
                if tracker_key not in active_trackers:
                    logger.warning(f"Tracker {tracker_key} not in memory")
                    continue
                active_tracker = active_trackers[tracker_key]

                is_involved = False
                for io_list in (inputs, outputs):
                    for io in io_list:
                        for amt in io.get('amount', []):
                            unit = amt.get('unit', '')
                            if unit != 'lovelace' and len(unit) >= 56 and unit[:56].lower() == tracker.policy_id.lower():
                                is_involved = True
                                break
                        if is_involved:
                            break
                    if is_involved:
                        break

                if is_involved:
                    analysis_data = {'inputs': inputs, 'outputs': outputs, 'tx': tx}
                    tx_type, ada_amount, token_amount, details = analyze_transaction_improved(analysis_data, tracker.policy_id)
                    details['hash'] = tx_hash
                    notification_data = {
                        'tx_details': analysis_data,
                        'policy_id': tracker.policy_id,
                        'ada_amount': ada_amount,
                        'token_amount': token_amount,
                        'analysis_details': {'type': tx_type, 'hash': tx_hash, **details},
                        'tracker': active_tracker
                    }
                    notification_queue.put(notification_data)
                    logger.info(f"Queued notification for {tracker.token_name} ({tx_hash})")

        return {"status": "success", "message": "Notifications queued"}
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@tasks.loop(seconds=1)
async def process_notification_queue():
    """Process queued notifications"""
    try:
        while not notification_queue.empty():
            data = notification_queue.get_nowait()
            try:
                await send_transaction_notification(
                    tracker=data['tracker'],
                    tx_type=data['analysis_details']['type'],
                    ada_amount=data['ada_amount'],
                    token_amount=data['token_amount'],
                    details=data['analysis_details']
                )
                notification_queue.task_done()
            except Exception as e:
                logger.error(f"Notification processing error: {str(e)}", exc_info=True)
                notification_queue.task_done()
    except Exception as e:
        logger.error(f"Queue loop error: {str(e)}", exc_info=True)

@bot.event
async def on_ready():
    """Bot startup"""
    try:
        if not process_notification_queue.is_running():
            process_notification_queue.start()
            logger.info("Started notification queue processor")

        trackers = database.get_trackers()
        for tracker in trackers:
            tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
            active_trackers[tracker_key] = TokenTracker(
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

        await bot.tree.sync()
        logger.info(f"Bot ready! Loaded {len(active_trackers)} trackers")
        for guild in bot.guilds:
            logger.info(f"Guild: {guild.name} (ID: {guild.id}), Channels: {[c.name for c in guild.channels]}")
    except Exception as e:
        logger.error(f"on_ready error: {str(e)}", exc_info=True)

def get_token_info(policy_id: str):
    """Get token metadata"""
    try:
        assets = api.assets_policy(policy_id)
        if not assets:
            return None
        asset = api.asset(assets[0].asset)
        metadata = vars(asset) if not isinstance(asset, Exception) else {}
        onchain_metadata = metadata.get('onchain_metadata', {})
        decimals = onchain_metadata.get('decimals', metadata.get('metadata', {}).get('decimals', 0))
        return {
            'asset': asset.asset,
            'policy_id': policy_id,
            'name': metadata.get('asset_name', 'Unknown'),
            'decimals': int(decimals),
            'metadata': onchain_metadata
        }
    except Exception as e:
        logger.error(f"Token info error for {policy_id}: {str(e)}", exc_info=True)
        return None

def format_token_amount(amount: int, decimals: int) -> str:
    """Format token amount with decimals"""
    if decimals == 0:
        return f"{amount:,}"
    formatted = amount / (10 ** decimals)
    return f"{formatted:,.{min(decimals, 6)}f}" if formatted < 1000 else f"{formatted:,.2f}"

def analyze_transaction_improved(tx_details, policy_id):
    """Analyze transaction"""
    try:
        inputs = tx_details.get('inputs', [])
        outputs = tx_details.get('outputs', [])
        token_info = get_token_info(policy_id)
        decimals = token_info.get('decimals', 0) if token_info else 0

        ada_in, ada_out, token_in, token_out = 0, 0, 0, 0
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

        ada_amount = abs(ada_out - ada_in) / 1_000_000
        token_amount = abs(token_out - token_in) / (10 ** decimals) if decimals else abs(token_out - token_in)
        if token_in and token_out:
            tx_type = 'wallet_transfer'
            token_amount = max([int(out.get('amount', [{}])[0].get('quantity', 0)) for out in outputs if out.get('amount', [{}])[0].get('unit', '').startswith(policy_id)]) / (10 ** decimals)
        elif token_in:
            tx_type = 'sell'
        elif token_out:
            tx_type = 'buy'
        else:
            tx_type = 'unknown'

        return tx_type, ada_amount, token_amount, {
            'ada_in': ada_in / 1_000_000, 'ada_out': ada_out / 1_000_000,
            'token_in': token_in, 'token_out': token_out,
            'raw_token_amount': token_amount * (10 ** decimals), 'decimals': decimals
        }
    except Exception as e:
        logger.error(f"Transaction analysis error: {str(e)}", exc_info=True)
        return 'unknown', 0, 0, {}

async def create_trade_embed(tx_details, policy_id, ada_amount, token_amount, tracker, analysis_details):
    """Create trade embed"""
    try:
        tx_type = analysis_details['type']
        embed = discord.Embed(
            title=f"{'💰' if tx_type == 'buy' else '💱'} Token {tx_type.capitalize()} Detected",
            description=f"Transaction Hash: [`{tx_details.get('hash', '')[:8]}...`](https://cardanoscan.io/transaction/{tx_details.get('hash', '')})",
            color=discord.Color.green() if tx_type == 'buy' else discord.Color.blue()
        )
        embed.add_field(
            name="📝 Overview",
            value=f"```\nType: DEX {tx_type.capitalize()}\nStatus: Confirmed\n```",
            inline=True
        )
        trade_info = (
            f"```\n{'ADA Spent' if tx_type == 'buy' else 'ADA Received'}: {ada_amount:,.2f}\n"
            f"{'Tokens Received' if tx_type == 'buy' else 'Tokens Sold'}: {format_token_amount(int(token_amount * (10 ** tracker.token_info.get('decimals', 0))), tracker.token_info.get('decimals', 0))}\n"
            f"Price/Token: {(ada_amount / token_amount):.6f}\n```"
        )
        embed.add_field(name="💰 Trade Details", value=trade_info, inline=True)
        embed.add_field(name="🔑 Policy ID", value=f"```{policy_id}```", inline=False)
        embed.set_author(name=tracker.token_name)
        if tracker.image_url:
            embed.set_thumbnail(url=tracker.image_url)
        embed.set_footer(text=f"Detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return embed
    except Exception as e:
        logger.error(f"Trade embed error: {str(e)}", exc_info=True)
        return None

async def create_transfer_embed(tx_details, policy_id, token_amount, tracker):
    """Create transfer embed"""
    try:
        embed = discord.Embed(
            title="↔️ Token Transfer Detected",
            description="Tokens transferred between wallets.",
            color=discord.Color.blue()
        )
        from_addr = tx_details.get('inputs', [{}])[0].get('address', 'Unknown')
        to_addr = tx_details.get('outputs', [{}])[0].get('address', 'Unknown')
        embed.add_field(
            name="🔄 Transfer Details",
            value=f"**From:** ```{from_addr[:20]}...{from_addr[-8:]}```\n**To:** ```{to_addr[:20]}...{to_addr[-8:]}```\n**Amount:** ```{format_token_amount(int(token_amount * 10**tracker.token_info.get('decimals', 0)), tracker.token_info.get('decimals', 0))}```",
            inline=False
        )
        embed.add_field(
            name="👤 Wallet Profiles",
            value=f"[Sender](https://cardanoscan.io/address/{from_addr})\n[Receiver](https://cardanoscan.io/address/{to_addr})",
            inline=False
        )
        embed.add_field(
            name="🔍 Transaction",
            value=f"[View](https://cardanoscan.io/transaction/{tx_details.get('hash', '')})",
            inline=False
        )
        if tracker.image_url:
            embed.set_thumbnail(url=tracker.image_url)
        embed.timestamp = discord.utils.utcnow()
        return embed
    except Exception as e:
        logger.error(f"Transfer embed error: {str(e)}", exc_info=True)
        return None

async def send_transaction_notification(tracker, tx_type, ada_amount, token_amount, details):
    """Send transaction notification"""
    try:
        if tx_type == 'wallet_transfer' and not tracker.track_transfers:
            return
        decimals = tracker.token_info.get('decimals', 0) if tracker.token_info else 0
        if token_amount < tracker.threshold:
            logger.info(f"Amount {token_amount:,.{decimals}f} below threshold {tracker.threshold}")
            return

        channel = bot.get_channel(tracker.channel_id)
        if not channel:
            logger.warning(f"Channel {tracker.channel_id} not cached, fetching...")
            try:
                channel = await bot.fetch_channel(tracker.channel_id)
            except discord.NotFound:
                logger.error(f"Channel {tracker.channel_id} not found")
                tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
                if tracker_key in active_trackers:
                    del active_trackers[tracker_key]
                database.delete_token_tracker(tracker.policy_id, tracker.channel_id)
                return
            except discord.Forbidden:
                logger.error(f"No access to channel {tracker.channel_id}")
                return

        embed = None
        if tx_type in ['buy', 'sell']:
            embed = await create_trade_embed(details, tracker.policy_id, ada_amount, token_amount, tracker, details)
            if embed:
                await channel.send(embed=embed)
                tracker.increment_trade_notifications()
        elif tx_type == 'wallet_transfer':
            embed = await create_transfer_embed(details, tracker.policy_id, token_amount, tracker)
            if embed:
                await channel.send(embed=embed)
                tracker.increment_transfer_notifications()
        logger.info(f"Sent {tx_type} notification to {tracker.channel_id}")
    except discord.Forbidden:
        logger.error(f"No permission for channel {tracker.channel_id}")
    except Exception as e:
        logger.error(f"Send notification error: {str(e)}", exc_info=True)

class TokenControls(discord.ui.View):
    def __init__(self, policy_id):
        super().__init__(timeout=None)
        self.policy_id = policy_id

    @discord.ui.button(label="Stop Tracking", style=discord.ButtonStyle.danger, emoji="⛔")
    async def stop_tracking(self, interaction: discord.Interaction, button: discord.ui.Button):
        tracker_key = f"{self.policy_id}:{interaction.channel_id}"
        if tracker_key in active_trackers:
            database.delete_token_tracker(self.policy_id, interaction.channel_id)
            del active_trackers[tracker_key]
            embed = discord.Embed(
                title="✅ Tracking Stopped",
                description=f"Stopped tracking: ```{self.policy_id}```",
                color=discord.Color.green()
            )
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(embed=embed, view=self)
            logger.info(f"Stopped tracking {self.policy_id}")
        else:
            await interaction.response.send_message("Not tracking this token.", ephemeral=True)

    @discord.ui.button(label="Toggle Transfers", style=discord.ButtonStyle.primary, emoji="🔄")
    async def toggle_transfers(self, interaction: discord.Interaction, button: discord.ui.Button):
        tracker_key = f"{self.policy_id}:{interaction.channel_id}"
        if tracker_key not in active_trackers:
            await interaction.response.send_message("Tracker not found.", ephemeral=True)
            return
        tracker = active_trackers[tracker_key]
        tracker.track_transfers = not tracker.track_transfers
        database.save_token_tracker({
            'policy_id': tracker.policy_id,
            'token_name': tracker.token_name,
            'image_url': tracker.image_url,
            'threshold': tracker.threshold,
            'channel_id': tracker.channel_id,
            'last_block': tracker.last_block,
            'track_transfers': tracker.track_transfers,
            'token_info': tracker.token_info
        })
        embed = discord.Embed(
            title="✅ Tracking Active",
            description=f"**Policy ID:** ```{tracker.policy_id}```\n**Name:** ```{tracker.token_name}```\n**Threshold:** ```{tracker.threshold:,.2f}```\n**Transfers:** ```{'Enabled' if tracker.track_transfers else 'Disabled'}```\n**Trades:** ```{tracker.trade_notifications}```\n**Transfers:** ```{tracker.transfer_notifications}```",
            color=discord.Color.blue()
        )
        if tracker.image_url:
            embed.set_thumbnail(url=tracker.image_url)
        await interaction.response.edit_message(embed=embed)

class TokenSetupModal(discord.ui.Modal, title="🪙 Token Setup"):
    def __init__(self):
        super().__init__()
        self.policy_id = discord.ui.TextInput(label="Policy ID", placeholder="Enter policy ID", required=True, min_length=56, max_length=56)
        self.token_name = discord.ui.TextInput(label="Token Name", placeholder="Enter token name", required=True)
        self.image_url = discord.ui.TextInput(label="Image URL", placeholder="Optional image URL", required=False)
        self.threshold = discord.ui.TextInput(label="Min Token Amount", placeholder="Default: 1000", required=False, default="1000")
        self.track_transfers = discord.ui.TextInput(label="Track Transfers", placeholder="yes/no (default: yes)", required=False, default="yes")
        self.add_item(self.policy_id)
        self.add_item(self.token_name)
        self.add_item(self.image_url)
        self.add_item(self.threshold)
        self.add_item(self.track_transfers)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            token_info = get_token_info(self.policy_id.value)
            if not token_info:
                await interaction.response.send_message("Invalid policy ID", ephemeral=True)
                return
            threshold = float(self.threshold.value) if self.threshold.value else 1000.0
            track_transfers = self.track_transfers.value.lower() != 'no'
            tracker = TokenTracker(
                policy_id=self.policy_id.value,
                channel_id=interaction.channel_id,
                token_name=self.token_name.value,
                image_url=self.image_url.value or None,
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
                title="✅ Tracking Started",
                description=f"**Policy ID:** ```{tracker.policy_id}```\n**Name:** ```{tracker.token_name}```\n**Threshold:** ```{tracker.threshold:,.2f}```\n**Transfers:** ```{'Enabled' if tracker.track_transfers else 'Disabled'}```",
                color=discord.Color.blue()
            )
            if tracker.image_url:
                embed.set_thumbnail(url=tracker.image_url)
            view = TokenControls(tracker.policy_id)
            await interaction.response.send_message(embed=embed, view=view)
            tracker_key = f"{tracker.policy_id}:{tracker.channel_id}"
            active_trackers[tracker_key] = tracker
            logger.info(f"Started tracking {tracker.token_name}")
        except Exception as e:
            logger.error(f"Setup error: {str(e)}", exc_info=True)
            await interaction.response.send_message("Setup failed", ephemeral=True)

@bot.tree.command(name="start", description="Start tracking a token")
async def start(interaction: discord.Interaction):
    await interaction.response.send_modal(TokenSetupModal())

@bot.tree.command(name="help", description="Show bot help")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Help",
        description="Track Cardano token transactions!",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="📝 Commands",
        value="**/start**: Start tracking\n**/status**: View trackers\n**/stop**: Stop all\n**/help**: This message",
        inline=False
    )
    embed.add_field(
        name="🔍 Features",
        value="• DEX Trades\n• Transfers\n• Real-time\n• Thresholds",
        inline=True
    )
    embed.add_field(
        name="🔔 Notifications",
        value="• Amounts\n• Addresses\n• Hash",
        inline=True
    )
    embed.set_footer(text="Contact admin for help.")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Show tracked tokens")
@app_commands.default_permissions(administrator=True)
async def status_command(interaction: discord.Interaction):
    channel_trackers = [t for t in active_trackers.values() if t.channel_id == interaction.channel_id]
    if not channel_trackers:
        await interaction.response.send_message("No trackers in this channel.", ephemeral=True)
        return
    for tracker in channel_trackers:
        embed = discord.Embed(
            title="✅ Tracking Active",
            description=f"**Policy ID:** ```{tracker.policy_id}```\n**Name:** ```{tracker.token_name}```\n**Threshold:** ```{tracker.threshold:,.2f}```\n**Transfers:** ```{'Enabled' if tracker.track_transfers else 'Disabled'}```\n**Trades:** ```{tracker.trade_notifications}```\n**Transfers:** ```{tracker.transfer_notifications}```",
            color=discord.Color.blue()
        )
        if tracker.image_url:
            embed.set_thumbnail(url=tracker.image_url)
        view = TokenControls(tracker.policy_id)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

@bot.tree.command(name="stop", description="Stop all trackers in channel")
@app_commands.default_permissions(administrator=True)
async def stop(interaction: discord.Interaction):
    channel_trackers = [t for t in active_trackers.values() if t.channel_id == interaction.channel_id]
    if not channel_trackers:
        await interaction.response.send_message("No trackers to stop.", ephemeral=True)
        return
    embed = discord.Embed(
        title="⚠️ Stop Tracking",
        description="Confirm stopping all trackers:",
        color=discord.Color.yellow()
    )
    embed.add_field(name="Tokens", value="\n".join([f"• {t.token_name} (`{t.policy_id}`)" for t in channel_trackers]), inline=False)

    class ConfirmView(discord.ui.View):
        @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
        async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
            database.remove_all_trackers_for_channel(interaction.channel_id)
            for key in list(active_trackers.keys()):
                if active_trackers[key].channel_id == interaction.channel_id:
                    del active_trackers[key]
            embed = discord.Embed(title="✅ Stopped", description="All trackers stopped.", color=discord.Color.green())
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(embed=embed, view=self)

        @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            embed = discord.Embed(title="❌ Cancelled", description="Tracking continues.", color=discord.Color.red())
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(embed=embed, view=self)

    await interaction.response.send_message(embed=embed, view=ConfirmView())

def run_webhook_server():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))

if __name__ == "__main__":
    threading.Thread(target=run_webhook_server, daemon=True).start()
    bot.run(os.getenv('DISCORD_TOKEN'))