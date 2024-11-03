from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base, engine, SessionLocal
from table_post import Post
from table_user import User

class Feed(Base):

    __tablename__ = "feed_action"
   # __table_args__ == { "schema" : "public"}
    user_id = Column(Integer, ForeignKey(User.id), primary_key=True)
    user = relationship("User")
    post_id = Column(Integer, ForeignKey(Post.id), primary_key=True)
    post = relationship("Post")
    action = Column(String)
    time = Column(DateTime)