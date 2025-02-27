# Cardano Token Purchase Tracker Discord Bot

A Discord bot that tracks token purchases on the Cardano blockchain using Blockfrost API.

## Features

- Track token purchases by Policy ID
- Set minimum ADA threshold for purchase detection
- Customizable project image in embeds
- Real-time purchase notifications
- Detailed transaction information including:
  - Buyer wallet address
  - Amount spent in ADA
  - Tokens received
  - Price per token

## Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with the following variables:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   BLOCKFROST_API_KEY=your_blockfrost_api_key
   ```

3. Create a Discord application and bot at https://discord.com/developers/applications
   - Enable necessary intents (Message Content Intent)
   - Copy the bot token to your `.env` file

4. Get a Blockfrost API key from https://blockfrost.io/
   - Create an account
   - Create a new project
   - Copy the API key to your `.env` file

5. Run the bot:
   ```bash
   python bot.py
   ```

## Usage

1. Invite the bot to your server with necessary permissions
2. Use the `/start` command with the following parameters:
   - `policy_id`: The Policy ID of the token to track
   - `image_url`: URL of the project image for embeds
   - `threshold`: Minimum ADA amount to trigger buy alerts

## Commands

- `/start` - Start tracking token purchases with specified parameters

## Example

```
/start policy_id:abc123... image_url:https://example.com/image.png threshold:5000
```
