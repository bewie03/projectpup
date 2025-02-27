from database import Database, Base
import os

def init_database():
    # Your PostgreSQL URL
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    
    print("Initializing database...")
    try:
        # Initialize database connection
        db = Database(DATABASE_URL)
        print("✅ Database connection successful!")
        
        # Test database operations
        print("\nTesting database operations...")
        
        # Test saving a token tracker
        test_tracker = {
            'policy_id': 'test_policy_123',
            'image_url': 'https://test.com/image.png',
            'threshold': 1000.0,
            'channel_id': 123456789,
            'last_block': None,
            'track_transfers': True
        }
        
        db.save_token_tracker(test_tracker)
        print("✅ Successfully saved test tracker")
        
        # Test retrieving trackers
        trackers = db.get_all_token_trackers()
        print(f"✅ Retrieved {len(trackers)} trackers from database")
        
        # Test updating last block
        db.update_last_block('test_policy_123', 123456789, 1000)
        print("✅ Successfully updated last block")
        
        # Test deleting tracker
        db.delete_token_tracker('test_policy_123', 123456789)
        print("✅ Successfully deleted test tracker")
        
        print("\nAll database operations completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    init_database()
