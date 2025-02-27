import os
from database import Base, Database
from sqlalchemy import create_engine

def migrate_database():
    # Get database URL
    database_url = os.getenv('DATABASE_URL')
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    # Create engine
    engine = create_engine(database_url)
    
    # Drop and recreate all tables
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    
    print("Database migration completed successfully!")

if __name__ == "__main__":
    migrate_database()
