from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, Boolean, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Create base class for declarative models
Base = declarative_base()

class TokenTracker(Base):
    """Model for token tracking configuration"""
    __tablename__ = 'token_trackers'
    
    id = Column(Integer, primary_key=True)
    policy_id = Column(String, nullable=False)
    token_name = Column(String, nullable=False)
    image_url = Column(String)
    threshold = Column(Float, default=0.0)
    channel_id = Column(BigInteger, nullable=False)
    last_block = Column(Integer, default=0)
    track_transfers = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    trade_notifications = Column(Integer, default=0)
    transfer_notifications = Column(Integer, default=0)
    
    def to_dict(self):
        """Convert tracker to dictionary"""
        return {
            'id': self.id,
            'policy_id': self.policy_id,
            'token_name': self.token_name,
            'image_url': self.image_url,
            'threshold': self.threshold,
            'channel_id': self.channel_id,
            'last_block': self.last_block,
            'track_transfers': self.track_transfers,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'trade_notifications': self.trade_notifications,
            'transfer_notifications': self.transfer_notifications
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
            
    def add_tracker(self, policy_id, token_name, channel_id, image_url=None, threshold=0.0):
        """Add a new token tracker"""
        session = self.Session()
        try:
            tracker = TokenTracker(
                policy_id=policy_id,
                token_name=token_name,
                channel_id=channel_id,
                image_url=image_url,
                threshold=threshold
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
            
    def remove_tracker(self, policy_id, channel_id):
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
