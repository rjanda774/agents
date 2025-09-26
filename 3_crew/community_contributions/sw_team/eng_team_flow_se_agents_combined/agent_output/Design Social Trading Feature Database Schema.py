```python
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, Index, UniqueConstraint
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.sql import func

# Define the base class for declarative class definitions
Base = declarative_base()

# Define the User table
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    followers = relationship('Follower', back_populates='user', cascade='all, delete-orphan')
    trades = relationship('TradeHistory', back_populates='user', cascade='all, delete-orphan')
    posts = relationship('CommunityPost', back_populates='user', cascade='all, delete-orphan')

# Define the Follower table
class Follower(Base):
    __tablename__ = 'followers'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    follower_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Relationships
    user = relationship('User', foreign_keys=[user_id], back_populates='followers')
    follower = relationship('User', foreign_keys=[follower_id])
    
    # Constraints
    __table_args__ = (UniqueConstraint('user_id', 'follower_id', name='_user_follower_uc'),)

# Define the TradeHistory table
class TradeHistory(Base):
    __tablename__ = 'trade_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    trade_details = Column(Text, nullable=False)
    trade_date = Column(DateTime, server_default=func.now())
    
    # Relationships
    user = relationship('User', back_populates='trades')

# Define the CommunityPost table
class CommunityPost(Base):
    __tablename__ = 'community_posts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    user = relationship('User', back_populates='posts')
    engagements = relationship('EngagementMetric', back_populates='post', cascade='all, delete-orphan')

# Define the EngagementMetric table
class EngagementMetric(Base):
    __tablename__ = 'engagement_metrics'
    
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey('community_posts.id', ondelete='CASCADE'), nullable=False)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    
    # Relationships
    post = relationship('CommunityPost', back_populates='engagements')

# Create an engine and session
def get_engine(user, password, host, port, db):
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, echo=True)

def create_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()

# Example usage
if __name__ == "__main__":
    # Replace with your actual database credentials
    engine = get_engine('your_user', 'your_password', 'localhost', '5432', 'your_database')
    Base.metadata.create_all(engine)
    session = create_session(engine)
```

This code defines a PostgreSQL database schema using SQLAlchemy for a social trading platform. It includes tables for users, followers, trade history, community posts, and engagement metrics. The code ensures proper normalization, indexing, and relationships with cascade behaviors. It also avoids hardcoded credentials by using a function to create the database engine.