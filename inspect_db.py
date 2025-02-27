import psycopg2
import os
import json

def get_db_connection():
    """Get a connection to the PostgreSQL database"""
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    return psycopg2.connect(DATABASE_URL)

def recreate_tables():
    """Drop and recreate the database tables"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Drop existing table
        cur.execute('DROP TABLE IF EXISTS trackers')
        
        # Create trackers table
        cur.execute('''
            CREATE TABLE trackers (
                policy_id TEXT PRIMARY KEY,
                channel_id BIGINT NOT NULL,
                token_name TEXT,
                image_url TEXT,
                threshold REAL NOT NULL DEFAULT 1000.0,
                track_transfers BOOLEAN NOT NULL DEFAULT TRUE,
                last_block BIGINT NOT NULL DEFAULT 0,
                trade_notifications INTEGER NOT NULL DEFAULT 0,
                transfer_notifications INTEGER NOT NULL DEFAULT 0,
                token_info JSONB
            )
        ''')
        
        conn.commit()
        print("Database tables recreated successfully!")
        
    except Exception as e:
        print(f"Error recreating tables: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def show_table_info():
    """Show detailed information about the database tables"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # List all tables
        print("\n=== Database Tables ===")
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name != 'pg_stat_statements_info'
        """)
        tables = cur.fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"\n=== Table: {table_name} ===")
            
            # Show columns and their types
            cur.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns = cur.fetchall()
            print("\nColumns:")
            for col in columns:
                print(f"  {col[0]:<20} {col[1]:<15} {'NULL' if col[2]=='YES' else 'NOT NULL':<10} Default: {col[3] or 'None'}")
            
            # Show row count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            print(f"\nTotal rows: {count}")
            
            if count > 0:
                # Show sample data
                cur.execute(f"SELECT * FROM {table_name} LIMIT 5")
                rows = cur.fetchall()
                print("\nSample data:")
                for row in rows:
                    print("\nRow:")
                    for i, value in enumerate(row):
                        if columns[i][0] == 'token_info' and value:
                            print(f"  {columns[i][0]}: {json.dumps(value, indent=2)}")
                        else:
                            print(f"  {columns[i][0]}: {value}")
            
            # Show indexes
            cur.execute("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = %s
            """, (table_name,))
            indexes = cur.fetchall()
            if indexes:
                print("\nIndexes:")
                for idx in indexes:
                    print(f"  {idx[0]}: {idx[1]}")
            
            print("-" * 80)
            
    except Exception as e:
        print(f"Error inspecting database: {str(e)}")
    finally:
        cur.close()
        conn.close()

def inspect_database():
    """Display all trackers in the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get all trackers
        cur.execute('SELECT * FROM trackers')
        trackers = cur.fetchall()
        
        print("\n=== Current Trackers ===")
        if not trackers:
            print("No trackers found in database")
        else:
            print(f"Found {len(trackers)} trackers:")
            for tracker in trackers:
                print(f"\nPolicy ID: {tracker[0]}")
                print(f"Channel ID: {tracker[1]}")
                print(f"Token Name: {tracker[2]}")
                print(f"Image URL: {tracker[3]}")
                print(f"Threshold: {tracker[4]}")
                print(f"Track Transfers: {tracker[5]}")
                print(f"Last Block: {tracker[6]}")
                print(f"Trade Notifications: {tracker[7]}")
                print(f"Transfer Notifications: {tracker[8]}")
                if tracker[9]:  # token_info
                    print(f"Token Info: {json.dumps(tracker[9], indent=2)}")
                print("-" * 50)
                
    except Exception as e:
        print(f"Error inspecting database: {str(e)}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # Show database info
    show_table_info()
    inspect_database()
