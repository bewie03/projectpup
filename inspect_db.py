import psycopg2
import os
import json

def get_db_connection():
    """Get a connection to the PostgreSQL database"""
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    return psycopg2.connect(DATABASE_URL)

def inspect_database():
    """Display all trackers in the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get all trackers
        cur.execute('SELECT * FROM trackers')
        rows = cur.fetchall()
        
        if not rows:
            print("No trackers found in database")
            return
            
        print("\nCurrent Trackers:")
        print("-----------------")
        
        for row in rows:
            print(f"\nPolicy ID: {row[0]}")
            print(f"Channel ID: {row[1]}")
            print(f"Token Name: {row[2]}")
            print(f"Image URL: {row[3]}")
            print(f"Threshold: {row[4]}")
            print(f"Track Transfers: {row[5]}")
            print(f"Last Block: {row[6]}")
            print(f"Trade Notifications: {row[7]}")
            print(f"Transfer Notifications: {row[8]}")
            if row[9]:  # token_info
                print(f"Token Info: {json.dumps(row[9], indent=2)}")
            print("-----------------")
            
    except Exception as e:
        print(f"Error inspecting database: {str(e)}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    inspect_database()
