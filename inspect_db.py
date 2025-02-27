import psycopg2

def inspect_database():
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # List all tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND table_name = 'token_trackers'
    """)
    tables = cur.fetchall()
    
    if not tables:
        print("No token_trackers table found!")
        return
        
    # Show token_trackers table info
    print("\n=== token_trackers table ===")
    
    # Get column info
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'token_trackers'
        ORDER BY ordinal_position
    """)
    columns = cur.fetchall()
    print("\nColumns:")
    for col in columns:
        print(f"{col[0]:<20} {col[1]}")
        
    # Get row count
    cur.execute("SELECT COUNT(*) FROM token_trackers")
    count = cur.fetchone()[0]
    print(f"\nTotal rows: {count}")
    
    # Show sample data
    if count > 0:
        cur.execute("SELECT * FROM token_trackers")
        rows = cur.fetchall()
        print("\nCurrent data:")
        for row in rows:
            print("\nToken tracker:")
            for i, col in enumerate(columns):
                print(f"{col[0]:<20} {row[i]}")
    
    cur.close()
    conn.close()

def update_database():
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    try:
        # Add missing columns if they don't exist
        cur.execute("""
            ALTER TABLE token_trackers
            ADD COLUMN IF NOT EXISTS image_url TEXT,
            ADD COLUMN IF NOT EXISTS last_block BIGINT DEFAULT 0,
            ADD COLUMN IF NOT EXISTS track_transfers BOOLEAN DEFAULT TRUE,
            ADD COLUMN IF NOT EXISTS trade_notifications INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS transfer_notifications INTEGER DEFAULT 0;
        """)

        # Update any existing rows to have default values
        cur.execute("""
            UPDATE token_trackers
            SET 
                last_block = COALESCE(last_block, 0),
                track_transfers = COALESCE(track_transfers, TRUE),
                trade_notifications = COALESCE(trade_notifications, 0),
                transfer_notifications = COALESCE(transfer_notifications, 0)
            WHERE last_block IS NULL 
               OR track_transfers IS NULL 
               OR trade_notifications IS NULL 
               OR transfer_notifications IS NULL;
        """)

        conn.commit()
        print("Successfully updated database schema and data")

    except Exception as e:
        conn.rollback()
        print(f"Error: {str(e)}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    print("Current database state:")
    inspect_database()
    
    print("\nUpdating database schema...")
    update_database()
    
    print("\nFinal database state:")
    inspect_database()
