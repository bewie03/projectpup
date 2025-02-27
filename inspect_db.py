import psycopg2
import os
import json

def get_db_connection():
    """Get a connection to the PostgreSQL database"""
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    return psycopg2.connect(DATABASE_URL)

def create_tables():
    """Create the database tables if they don't exist"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Create trackers table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS trackers (
                policy_id TEXT PRIMARY KEY,
                channel_id INTEGER,
                token_name TEXT,
                image_url TEXT,
                threshold REAL,
                track_transfers BOOLEAN,
                last_block INTEGER,
                trade_notifications INTEGER,
                transfer_notifications INTEGER,
                token_info JSONB
            )
        ''')
        
        conn.commit()
        print("Database tables created successfully!")
        
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def migrate_database():
    """Add token_info column to token_trackers table"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Check if token_info column exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'trackers' 
            AND column_name = 'token_info'
        """)
        exists = cur.fetchone()
        
        if not exists:
            print("Adding token_info column to trackers table...")
            cur.execute('ALTER TABLE trackers ADD COLUMN token_info JSONB')
            conn.commit()
            print("Migration successful!")
        else:
            print("token_info column already exists")
            
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def inspect_database():
    """Display all trackers in the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get column names
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'trackers'
            ORDER BY ordinal_position
        """)
        columns = [col[0] for col in cur.fetchall()]
        
        # Get all trackers
        cur.execute('SELECT * FROM trackers')
        rows = cur.fetchall()
        
        if not rows:
            print("No trackers found in database")
            return
            
        print("\nCurrent trackers in database:")
        print("-" * 80)
        
        for row in rows:
            print("\nTracker:")
            for i, value in enumerate(row):
                if columns[i] == 'token_info' and value:
                    # Pretty print JSON
                    print(f"{columns[i]}: {json.dumps(value, indent=2)}")
                else:
                    print(f"{columns[i]}: {value}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error inspecting database: {str(e)}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # First create/migrate the database
    create_tables()
    migrate_database()
    
    # Then show current state
    inspect_database()
