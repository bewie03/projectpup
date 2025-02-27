import os
from sqlalchemy import create_engine, text

def migrate():
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

    # Create engine
    engine = create_engine(database_url)

    # Add new columns if they don't exist
    with engine.connect() as conn:
        # Check if columns exist
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='token_trackers' 
            AND column_name IN ('trade_notifications', 'transfer_notifications');
        """))
        existing_columns = [row[0] for row in result]

        # Add trade_notifications if it doesn't exist
        if 'trade_notifications' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE token_trackers 
                ADD COLUMN trade_notifications INTEGER DEFAULT 0;
            """))
            print("Added trade_notifications column")

        # Add transfer_notifications if it doesn't exist
        if 'transfer_notifications' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE token_trackers 
                ADD COLUMN transfer_notifications INTEGER DEFAULT 0;
            """))
            print("Added transfer_notifications column")

        conn.commit()

if __name__ == "__main__":
    migrate()
