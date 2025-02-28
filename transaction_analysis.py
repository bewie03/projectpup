import logging

# Get logger
logger = logging.getLogger(__name__)

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
        from bot import get_token_info  # Import here to avoid circular import
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
