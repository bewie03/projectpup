import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def migrate_database():
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    
    # Convert postgres:// to postgresql:// if needed
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    print("Dropping existing tables...")
    cur.execute("""
        DROP TABLE IF EXISTS trackers;
    """)
    
    print("Creating tables...")
    cur.execute("""
        CREATE TABLE trackers (
            policy_id VARCHAR(56),
            channel_id BIGINT,
            token_name VARCHAR(255),
            image_url VARCHAR(255),
            threshold FLOAT NOT NULL DEFAULT 1000,
            track_transfers BOOLEAN NOT NULL DEFAULT true,
            last_block BIGINT NOT NULL DEFAULT 0,
            trade_notifications INTEGER NOT NULL DEFAULT 0,
            transfer_notifications INTEGER NOT NULL DEFAULT 0,
            token_info JSONB,
            PRIMARY KEY (policy_id, channel_id)
        );
    """)
    
    print("Migration completed successfully!")
    cur.close()
    conn.close()

if __name__ == "__main__":
    migrate_database()
