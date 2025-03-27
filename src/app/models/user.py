from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from src.app import db


class User(db.Model, UserMixin):
    """
    User model for authentication and basic user information.
    
    This class extends Flask-Login's UserMixin to provide default
    implementations of is_authenticated(), is_active(), etc.
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')  # student, faculty, admin
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationship
    student = db.relationship('Student', back_populates='user', uselist=False, lazy='joined')
    faculty = db.relationship('Faculty', back_populates='user', uselist=False, lazy='joined')
    
    @property
    def password(self):
        """
        Prevent password from being accessed directly
        """
        raise AttributeError('password is not a readable attribute')
    
    @password.setter
    def password(self, password):
        """
        Set password to a hashed value
        """
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password):
        """
        Check if hashed password matches user password
        """
        return check_password_hash(self.password_hash, password)
    
    @property
    def name(self):
        """
        Returns the full name of the user
        """
        return f"{self.first_name} {self.last_name}"
    
    def update_last_login(self):
        """
        Update the last login timestamp
        """
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def is_admin(self):
        """
        Check if user has admin role
        """
        return self.role == 'admin'
    
    def is_faculty(self):
        """
        Check if user has faculty role
        """
        return self.role == 'faculty'
    
    def is_student(self):
        """
        Check if user has student role
        """
        return self.role == 'student'
    
    def to_dict(self):
        """
        Convert user object to dictionary
        """
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'name': self.name,
            'role': self.role,
            'active': self.active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.email}>'