import os
import logging
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Create SQLAlchemy base class
Base = declarative_base()

class TokenTracker(Base):
    """Database model for token tracking configuration"""
    __tablename__ = 'token_trackers'

    id = Column(Integer, primary_key=True)
    policy_id = Column(String, nullable=False)
    token_name = Column(String, nullable=False)
    image_url = Column(String)
    threshold = Column(Float, nullable=False)
    channel_id = Column(BigInteger, nullable=False)
    last_block = Column(Integer)
    track_transfers = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert model to dictionary for TokenTracker class"""
        return {
            'policy_id': self.policy_id,
            'token_name': self.token_name,
            'image_url': self.image_url,
            'threshold': self.threshold,
            'channel_id': self.channel_id,
            'last_block': self.last_block,
            'track_transfers': self.track_transfers
        }

class Database:
    """Database connection and operations handler"""
    def __init__(self, database_url=None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if self.database_url.startswith('postgres://'):
            self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)
        
        try:
            self.engine = create_engine(self.database_url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}", exc_info=True)
            raise

    def save_token_tracker(self, tracker_data):
        """Save or update a token tracker configuration"""
        session = self.Session()
        try:
            # Check if tracker already exists
            existing = session.query(TokenTracker).filter_by(
                policy_id=tracker_data['policy_id'],
                channel_id=tracker_data['channel_id']
            ).first()

            if existing:
                # Update existing tracker
                for key, value in tracker_data.items():
                    setattr(existing, key, value)
                tracker = existing
            else:
                # Create new tracker
                tracker = TokenTracker(**tracker_data)
                session.add(tracker)

            session.commit()
            logger.info(f"Saved token tracker for policy_id: {tracker_data['policy_id']}")
            return tracker.to_dict()

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error while saving token tracker: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()

    def get_all_token_trackers(self):
        """Retrieve all token trackers"""
        session = self.Session()
        try:
            trackers = session.query(TokenTracker).all()
            return [tracker.to_dict() for tracker in trackers]
        except SQLAlchemyError as e:
            logger.error(f"Database error while retrieving token trackers: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()

    def update_last_block(self, policy_id, channel_id, block_height):
        """Update the last processed block height for a token tracker"""
        session = self.Session()
        try:
            tracker = session.query(TokenTracker).filter_by(
                policy_id=policy_id,
                channel_id=channel_id
            ).first()
            
            if tracker:
                tracker.last_block = block_height
                session.commit()
                logger.info(f"Updated last block height for policy_id {policy_id}: {block_height}")
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error while updating last block: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()

    def delete_token_tracker(self, policy_id, channel_id):
        """Delete a token tracker configuration"""
        session = self.Session()
        try:
            tracker = session.query(TokenTracker).filter_by(
                policy_id=policy_id,
                channel_id=channel_id
            ).first()
            
            if tracker:
                session.delete(tracker)
                session.commit()
                logger.info(f"Deleted token tracker for policy_id: {policy_id}")
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error while deleting token tracker: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
