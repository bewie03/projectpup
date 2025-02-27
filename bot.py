import os
import discord
from discord import app_commands
from discord.ext import commands, tasks
from blockfrost import BlockFrostApi, ApiError, ApiUrls
from dotenv import load_dotenv
import asyncio
from datetime import datetime, timezone
import json

# Load environment variables
load_dotenv()

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

async def get_token_info(api: BlockFrostApi, policy_id: str):
    try:
        # Get all assets under the policy
        assets = await api.assets_policy(policy_id=policy_id)
        if not assets:
            return None
        
        # Get detailed info for the first asset (main token)
        asset_info = await api.asset(assets[0].asset)
        return asset_info
    except ApiError as e:
        print(f"Error fetching token info: {e}")
        return None

async def get_transaction_details(api: BlockFrostApi, tx_hash: str):
    try:
        # Get detailed transaction information
        tx = await api.transaction(tx_hash)
        
        # Get transaction UTXOs
        utxos = await api.transaction_utxos(tx_hash)
        
        # Get transaction metadata if available
        metadata = await api.transaction_metadata(tx_hash)
        
        return tx, utxos, metadata
    except ApiError as e:
        print(f"Error fetching transaction details: {e}")
        return None, None, None

@bot.event
async def on_ready():
    print(f'Bot is ready: {bot.user}')
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
        check_transactions.start()
    except Exception as e:
        print(f"Error syncing commands: {e}")

@bot.tree.command(name="start", description="Start tracking token purchases")
async def start(interaction: discord.Interaction, policy_id: str, image_url: str, threshold: float):
    try:
        # Create embed for confirmation
        embed = discord.Embed(
            title="🚀 Token Tracking Initialized",
            description="Successfully started monitoring token purchases on the Cardano blockchain.",
            color=discord.Color.brand_green()
        )
        
        # Add project details fields
        embed.add_field(
            name="📋 Token Policy ID",
            value=f"`{policy_id}`",
            inline=False
        )
        embed.add_field(
            name="💎 Transaction Threshold",
            value=f"```{threshold:,.2f} ADA```",
            inline=False
        )
        
        # Add footer with timestamp
        embed.set_footer(text="Tracking started at")
        embed.timestamp = discord.utils.utcnow()
        
        # Set project image
        embed.set_thumbnail(url=image_url)
        
        # Add setup confirmation
        embed.add_field(
            name="✅ Status",
            value="Bot is now actively monitoring transactions",
            inline=False
        )

        # Store tracking configuration
        active_trackers[policy_id] = TokenTracker(
            policy_id=policy_id,
            image_url=image_url,
            threshold=threshold,
            channel_id=interaction.channel_id
        )

        await interaction.response.send_message(embed=embed)

    except Exception as e:
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
        error_embed = discord.Embed(
            title="❌ Error Fetching Volume Statistics",
            description=f"```{str(e)}```",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=error_embed, ephemeral=True)

@tasks.loop(seconds=60)
async def check_transactions():
    try:
        api = BlockFrostApi(
            project_id=os.getenv('BLOCKFROST_API_KEY'),
            base_url=ApiUrls.mainnet.value
        )
        
        for policy_id, tracker in active_trackers.items():
            # Get latest block if not set
            if not tracker.last_block:
                latest_block = await api.block_latest()
                tracker.last_block = latest_block.height
                continue

            # Get transactions since last check
            transactions = await api.address_transactions(policy_id, from_block=tracker.last_block)
            
            for tx in transactions:
                # Analyze transaction
                tx_details = await api.transaction(tx.tx_hash)
                
                # Check if transaction meets threshold
                ada_amount = sum(
                    output.amount[0].quantity / 1_000_000 
                    for output in tx_details.outputs 
                    if output.amount[0].unit == "lovelace"
                )
                
                if ada_amount >= tracker.threshold:
                    # Get token details
                    token_amount = next(
                        (
                            int(amount.quantity)
                            for output in tx_details.outputs
                            for amount in output.amount
                            if amount.unit.startswith(policy_id)
                        ),
                        0
                    )
                    
                    if token_amount > 0:
                        # Calculate price per token
                        price_per_token = ada_amount / token_amount
                        
                        # Create buy alert embed with enhanced information
                        embed = discord.Embed(
                            title="🌟 New Token Purchase Alert",
                            description=f"A significant token purchase has been detected on the Cardano blockchain.",
                            color=discord.Color.brand_green()
                        )
                        
                        # Transaction Overview
                        overview = (
                            f"• **Size:** Large Purchase\n"
                            f"• **Status:** Confirmed\n"
                            f"• **Network:** Cardano Mainnet"
                        )
                        embed.add_field(
                            name="📝 Transaction Overview",
                            value=overview,
                            inline=False
                        )
                        
                        # Buyer Details
                        buyer_address = tx_details.inputs[0].address
                        embed.add_field(
                            name="👤 Buyer Details",
                            value=(
                                f"**Wallet:** ```{buyer_address[:20]}...{buyer_address[-8:]}```\n"
                                f"**Profile:** [View on Cardanoscan](https://cardanoscan.io/address/{buyer_address})"
                            ),
                            inline=False
                        )
                        
                        # Purchase Information
                        purchase_info = (
                            f"**Amount Spent:** ```{ada_amount:,.2f} ADA```\n"
                            f"**Tokens Received:** ```{token_amount:,}```\n"
                            f"**Price per Token:** ```{price_per_token:.6f} ADA```"
                        )
                        embed.add_field(
                            name="💰 Purchase Information",
                            value=purchase_info,
                            inline=False
                        )
                        
                        # Market Impact (if we have 24h data)
                        if tracker.total_volume_24h > 0:
                            volume_percentage = (ada_amount / tracker.total_volume_24h) * 100
                            market_impact = (
                                f"**24h Volume:** ```{tracker.total_volume_24h:,.2f} ADA```\n"
                                f"**% of 24h Volume:** ```{volume_percentage:.2f}%```\n"
                                f"**24h Transactions:** ```{tracker.transactions_24h}```"
                            )
                            embed.add_field(
                                name="📊 Market Impact",
                                value=market_impact,
                                inline=False
                            )
                        
                        # Transaction Links
                        tx_links = (
                            f"🔍 [View Transaction](https://cardanoscan.io/transaction/{tx.tx_hash})\n"
                            f"📈 [View Token](https://cardanoscan.io/token/{policy_id})"
                        )
                        embed.add_field(
                            name="🔗 Transaction Links",
                            value=tx_links,
                            inline=False
                        )
                        
                        # Set thumbnail and footer
                        embed.set_thumbnail(url=tracker.image_url)
                        embed.timestamp = discord.utils.utcnow()
                        embed.set_footer(
                            text=f"Transaction detected at • Block #{tx_details.block_height}",
                            icon_url="https://cardanoscan.io/images/favicon.ico"
                        )
                        
                        # Send alert to channel
                        channel = bot.get_channel(tracker.channel_id)
                        if channel:
                            await channel.send(embed=embed)
            
            # Update last checked block
            latest_block = await api.block_latest()
            tracker.last_block = latest_block.height

    except Exception as e:
        print(f"Error checking transactions: {e}")

# Run the bot
if __name__ == "__main__":
    bot.run(os.getenv('DISCORD_TOKEN'))
