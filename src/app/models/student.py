from datetime import datetime
from src.app import db
from sqlalchemy.ext.hybrid import hybrid_property


class Student(db.Model):
    """
    Student model for storing student-specific information and academic records.
    
    This model extends the User model with additional fields for academic status
    prediction and tracking student performance.
    """
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False)  # University-assigned student ID
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=True)  # Current course/program
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=True)
    
    # Demographic information
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    marital_status = db.Column(db.String(20), nullable=True)  # Single, Married, etc.
    nationality = db.Column(db.String(50), nullable=True)
    
    # Academic background
    previous_qualification = db.Column(db.String(100), nullable=True)  # Previous education level
    application_mode = db.Column(db.String(50), nullable=True)  # How they applied
    application_order = db.Column(db.Integer, nullable=True)  # Priority of application
    attendance_type = db.Column(db.String(20), nullable=True)  # Daytime/evening attendance
    
    # Socioeconomic indicators
    displaced = db.Column(db.Boolean, default=False)  # Whether the student is from another region
    special_needs = db.Column(db.Boolean, default=False)  # Educational special needs
    debtor = db.Column(db.Boolean, default=False)  # Has unpaid fees
    tuition_fees_up_to_date = db.Column(db.Boolean, default=True)  # Tuition payment status
    scholarship_holder = db.Column(db.Boolean, default=False)  # Has scholarship
    international = db.Column(db.Boolean, default=False)  # International student
    
    # Family background
    mother_qualification = db.Column(db.String(100), nullable=True)
    father_qualification = db.Column(db.String(100), nullable=True)
    mother_occupation = db.Column(db.String(100), nullable=True)
    father_occupation = db.Column(db.String(100), nullable=True)
    
    # Academic performance tracking (1st semester)
    sem1_credits = db.Column(db.Float, nullable=True)  # Curricular units credited
    sem1_enrolled = db.Column(db.Float, nullable=True)  # Curricular units enrolled
    sem1_evaluations = db.Column(db.Float, nullable=True)  # Number of evaluations
    sem1_approved = db.Column(db.Float, nullable=True)  # Curricular units approved
    sem1_grade = db.Column(db.Float, nullable=True)  # Grade (0-20 scale)
    sem1_without_evaluations = db.Column(db.Float, nullable=True)  # Units without evaluations
    
    # Academic performance tracking (2nd semester)
    sem2_credits = db.Column(db.Float, nullable=True)
    sem2_enrolled = db.Column(db.Float, nullable=True)
    sem2_evaluations = db.Column(db.Float, nullable=True)
    sem2_approved = db.Column(db.Float, nullable=True)
    sem2_grade = db.Column(db.Float, nullable=True)
    sem2_without_evaluations = db.Column(db.Float, nullable=True)
    
    # Economic context
    unemployment_rate = db.Column(db.Float, nullable=True)  # Regional unemployment rate
    inflation_rate = db.Column(db.Float, nullable=True)  # Inflation rate
    gdp = db.Column(db.Float, nullable=True)  # GDP
    
    # Prediction and status tracking
    current_status = db.Column(db.String(20), nullable=True)  # Enrolled, Graduated, Dropout
    risk_level = db.Column(db.String(20), nullable=True)  # Low, Medium, High
    risk_score = db.Column(db.Float, nullable=True)  # Numerical risk score (0-1)
    last_prediction_date = db.Column(db.DateTime, nullable=True)
    
    # Additional fields
    notes = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', back_populates='student')
    course = db.relationship('Course', back_populates='students')
    department = db.relationship('Department', back_populates='students')
    predictions = db.relationship('Prediction', back_populates='student', lazy='dynamic')
    interventions = db.relationship('Intervention', back_populates='student', lazy='dynamic')
    
    @hybrid_property
    def overall_grade(self):
        """Calculate overall grade across both semesters"""
        grades = []
        weights = []
        
        if self.sem1_grade is not None and self.sem1_enrolled is not None and self.sem1_enrolled > 0:
            grades.append(self.sem1_grade)
            weights.append(self.sem1_enrolled)
            
        if self.sem2_grade is not None and self.sem2_enrolled is not None and self.sem2_enrolled > 0:
            grades.append(self.sem2_grade)
            weights.append(self.sem2_enrolled)
        
        if not grades:
            return None
            
        return sum(g * w for g, w in zip(grades, weights)) / sum(weights)
    
    @hybrid_property
    def overall_success_rate(self):
        """Calculate overall success rate (approved / enrolled)"""
        enrolled = 0
        approved = 0
        
        if self.sem1_enrolled is not None and self.sem1_approved is not None:
            enrolled += self.sem1_enrolled
            approved += self.sem1_approved
            
        if self.sem2_enrolled is not None and self.sem2_approved is not None:
            enrolled += self.sem2_enrolled
            approved += self.sem2_approved
            
        if enrolled == 0:
            return 0
            
        return (approved / enrolled) * 100
    
    @hybrid_property
    def engagement_score(self):
        """Calculate student engagement score based on evaluations and attendance"""
        if not self.sem1_enrolled and not self.sem2_enrolled:
            return 0
            
        total_enrolled = (self.sem1_enrolled or 0) + (self.sem2_enrolled or 0)
        total_evals = (self.sem1_evaluations or 0) + (self.sem2_evaluations or 0)
        total_without_evals = (self.sem1_without_evaluations or 0) + (self.sem2_without_evaluations or 0)
        
        if total_enrolled == 0:
            return 0
            
        # Base score from evaluation participation
        eval_ratio = total_evals / (total_enrolled * 2)  # Assuming 2 evaluations per unit on average
        attendance_ratio = 1 - (total_without_evals / total_enrolled)
        
        # Combine with other factors
        engagement = (eval_ratio * 60) + (attendance_ratio * 40)
        
        # Adjust for debt and tuition status
        if self.debtor:
            engagement *= 0.9
        if not self.tuition_fees_up_to_date:
            engagement *= 0.95
            
        return min(round(engagement), 100)  # Cap at 100
    
    @property
    def risk_percentage(self):
        """Convert risk score to percentage"""
        if self.risk_score is None:
            return None
        return round(self.risk_score * 100)
    
    @property
    def overall_percentage(self):
        """Convert overall grade to percentage (based on 20-point scale)"""
        if self.overall_grade is None:
            return None
        return round((self.overall_grade / 20) * 100)
    
    def get_latest_prediction(self):
        """Get the latest prediction for this student"""
        return self.predictions.order_by(Prediction.created_at.desc()).first()
    
    def update_risk_from_prediction(self, prediction):
        """Update risk metrics based on a new prediction"""
        self.risk_score = prediction.risk_score
        
        # Set risk level based on score
        if prediction.risk_score < 0.3:
            self.risk_level = 'Low'
        elif prediction.risk_score < 0.7:
            self.risk_level = 'Medium'
        else:
            self.risk_level = 'High'
            
        self.last_prediction_date = prediction.created_at
        self.current_status = prediction.predicted_status
        db.session.commit()
    
    def to_dict(self):
        """Convert student object to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'student_id': self.student_id,
            'course_id': self.course_id,
            'department_id': self.department_id,
            'name': self.user.name if self.user else None,
            'email': self.user.email if self.user else None,
            'age': self.age,
            'gender': self.gender,
            'marital_status': self.marital_status,
            'nationality': self.nationality,
            'current_status': self.current_status,
            'risk_level': self.risk_level,
            'risk_score': self.risk_score,
            'risk_percentage': self.risk_percentage,
            'overall_grade': self.overall_grade,
            'overall_percentage': self.overall_percentage,
            'overall_success_rate': self.overall_success_rate,
            'engagement_score': self.engagement_score,
            'scholarship_holder': self.scholarship_holder,
            'international': self.international,
            'special_needs': self.special_needs,
            'last_prediction_date': self.last_prediction_date.isoformat() if self.last_prediction_date else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
    
    def __repr__(self):
        return f'<Student {self.student_id}>'