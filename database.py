from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, Boolean, DateTime, text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os
import logging
from datetime import datetime
import json

# Set up logging
logger = logging.getLogger(__name__)

# Create base class for declarative models
Base = declarative_base()

class TokenTracker(Base):
    """Model for token tracking configuration"""
    __tablename__ = 'trackers'
    
    policy_id = Column(String, primary_key=True)
    channel_id = Column(BigInteger, nullable=False)
    token_name = Column(String)
    image_url = Column(String)
    threshold = Column(Float, default=1000.0, nullable=False)
    track_transfers = Column(Boolean, default=True, nullable=False)
    last_block = Column(BigInteger, default=0, nullable=False)
    trade_notifications = Column(Integer, default=0, nullable=False)
    transfer_notifications = Column(Integer, default=0, nullable=False)
    token_info = Column(JSON)
    
    def to_dict(self):
        """Convert tracker to dictionary"""
        return {
            'policy_id': self.policy_id,
            'channel_id': self.channel_id,
            'token_name': self.token_name,
            'image_url': self.image_url,
            'threshold': self.threshold,
            'track_transfers': self.track_transfers,
            'last_block': self.last_block,
            'trade_notifications': self.trade_notifications,
            'transfer_notifications': self.transfer_notifications,
            'token_info': self.token_info
        }

class Database:
    """Database connection and operations handler"""
    def __init__(self, database_url=None):
        """Initialize database connection"""
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("No database URL provided")
            
        # Handle Heroku's updated postgres:// to postgresql:// URL format
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
            
        # Create engine and session
        self.engine = create_engine(self.database_url)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        self.Session = sessionmaker(bind=self.engine)
        
    def get_trackers(self):
        """Get all token trackers from the database"""
        session = self.Session()
        try:
            trackers = session.query(TokenTracker).all()
            return trackers
        except SQLAlchemyError as e:
            logger.error(f"Database error while retrieving token trackers: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
            
    def get_tracker(self, policy_id, channel_id):
        """Get a specific token tracker"""
        session = self.Session()
        try:
            tracker = session.query(TokenTracker).filter_by(
                policy_id=policy_id,
                channel_id=channel_id
            ).first()
            return tracker
        except SQLAlchemyError as e:
            logger.error(f"Database error while retrieving token tracker: {str(e)}", exc_info=True)
            return None
        finally:
            session.close()
            
    def add_tracker(self, policy_id, token_name, channel_id, image_url=None, threshold=1000.0, token_info=None):
        """Add a new token tracker"""
        session = self.Session()
        try:
            tracker = TokenTracker(
                policy_id=policy_id,
                token_name=token_name,
                channel_id=channel_id,
                image_url=image_url,
                threshold=threshold,
                token_info=token_info
            )
            session.add(tracker)
            session.commit()
            return tracker
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error while adding token tracker: {str(e)}", exc_info=True)
            return None
        finally:
            session.close()
            
    def delete_token_tracker(self, policy_id, channel_id):
        """Remove a token tracker"""
        session = self.Session()
        try:
            tracker = session.query(TokenTracker).filter_by(
                policy_id=policy_id,
                channel_id=channel_id
            ).first()
            if tracker:
                session.delete(tracker)
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            logger.error(f"Database error while removing token tracker: {str(e)}", exc_info=True)
            session.rollback()
            return False
        finally:
            session.close()

    def remove_all_trackers_for_channel(self, channel_id):
        """Remove all token trackers for a specific channel"""
        session = self.Session()
        try:
            trackers = session.query(TokenTracker).filter_by(channel_id=channel_id).all()
            for tracker in trackers:
                session.delete(tracker)
            session.commit()
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database error while removing all token trackers for channel: {str(e)}", exc_info=True)
            session.rollback()
            return False
        finally:
            session.close()

    def save_token_tracker(self, tracker_data):
        """Save a token tracker to the database"""
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
            return tracker
        except SQLAlchemyError as e:
            logger.error(f"Database error while saving token tracker: {str(e)}", exc_info=True)
            session.rollback()
            return None
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
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error while updating last block: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
            
    def increment_notification_count(self, policy_id, channel_id, notification_type):
        """Increment the notification count for a token tracker"""
        session = self.Session()
        try:
            tracker = session.query(TokenTracker).filter_by(
                policy_id=policy_id,
                channel_id=channel_id
            ).first()
            if tracker:
                if notification_type == 'trade':
                    tracker.trade_notifications += 1
                elif notification_type == 'transfer':
                    tracker.transfer_notifications += 1
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error while incrementing notification count: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
            
    def get_all_token_trackers(self):
        """Retrieve all token trackers from the database.
        
        Returns:
            list: A list of all token trackers in the database
        """
        try:
            session = self.Session()
            trackers = session.query(TokenTracker).all()
            return trackers
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving token trackers: {str(e)}")
            return []
        finally:
            session.close()

# Create database instance
database = Database()

# Export the database instance
__all__ = ['database', 'TokenTracker', 'Database']
