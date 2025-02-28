import logging
import discord
from datetime import datetime

# Get logger
logger = logging.getLogger(__name__)

async def send_transaction_notification(tracker, tx_type, ada_amount, token_amount, details):
    """Send a notification about a transaction to the appropriate Discord channel"""
    try:
        # Import here to avoid circular imports
        from bot import bot, create_trade_embed, create_transfer_embed
        
        # Skip unknown transactions
        if tx_type == 'unknown':
            logger.info("Skipping notification for unknown transaction type")
            return

        # Get the channel
        channel = bot.get_channel(tracker.channel_id)
        if not channel:
            logger.error(f"Could not find channel {tracker.channel_id} for token {tracker.token_name}. The bot may have been removed from the server or the channel may have been deleted.")
            # TODO: Consider removing or marking this tracker as inactive
            return

        # Skip if amount is below threshold
        if token_amount < tracker.threshold:
            logger.info(f"Transaction amount {token_amount} below threshold {tracker.threshold}")
            return

        # Create appropriate embed based on transaction type
        if tx_type in ['buy', 'sell']:
            # Create trade embed
            embed = await create_trade_embed(details, tracker.policy_id, ada_amount, token_amount, tracker, details)
            if embed:
                await channel.send(embed=embed)
                tracker.increment_trade_notifications()
                logger.info(f"Sent trade notification to channel {channel.name}")
        elif tx_type == 'wallet_transfer' and tracker.track_transfers:
            # Create transfer embed
            embed = await create_transfer_embed(details, tracker.policy_id, token_amount, tracker)
            if embed:
                await channel.send(embed=embed)
                tracker.increment_transfer_notifications()
                logger.info(f"Sent transfer notification to channel {channel.name}")
        else:
            logger.info(f"Skipping notification for {tx_type} (transfer tracking: {tracker.track_transfers})")

    except Exception as e:
        logger.error(f"Error sending transaction notification: {str(e)}", exc_info=True)
