from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, TextAreaField, PasswordField, BooleanField, SelectField
from wtforms import IntegerField, DateField, TimeField, FloatField, HiddenField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError, Optional, NumberRange
from datetime import datetime, date

class ContactForm(FlaskForm):
    """Form for contact page"""
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=100)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    subject = StringField('Subject', validators=[DataRequired(), Length(min=5, max=100)])
    message = TextAreaField('Message', validators=[DataRequired(), Length(min=10, max=2000)])
    submit = SubmitField('Send Message')

class UserProfileForm(FlaskForm):
    """Form for editing user profile"""
    first_name = StringField('First Name', validators=[DataRequired(), Length(min=2, max=50)])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone Number', validators=[Optional(), Length(max=20)])
    profile_picture = FileField('Update Profile Picture', validators=[
        FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')
    ])
    bio = TextAreaField('Bio', validators=[Optional(), Length(max=500)])
    submit = SubmitField('Update Profile')

class ChangePasswordForm(FlaskForm):
    """Form for changing password"""
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[
        DataRequired(),
        Length(min=8, message='Password must be at least 8 characters long')
    ])
    confirm_password = PasswordField('Confirm New Password', validators=[
        DataRequired(),
        EqualTo('new_password', message='Passwords must match')
    ])
    submit = SubmitField('Change Password')

class InterventionForm(FlaskForm):
    """Form for creating/editing interventions"""
    student_id = SelectField('Student', coerce=int, validators=[DataRequired()])
    type = SelectField('Intervention Type', validators=[DataRequired()], choices=[
        ('academic_support', 'Academic Support'),
        ('counseling', 'Counseling'),
        ('financial_aid', 'Financial Aid'),
        ('study_skills', 'Study Skills Development'),
        ('mentoring', 'Mentoring Program'),
        ('custom', 'Custom Intervention')
    ])
    description = TextAreaField('Description', validators=[DataRequired(), Length(min=10, max=1000)])
    start_date = DateField('Start Date', validators=[DataRequired()], format='%Y-%m-%d', default=date.today)
    end_date = DateField('End Date', validators=[Optional()], format='%Y-%m-%d')
    follow_up_date = DateField('Follow-up Date', validators=[DataRequired()], format='%Y-%m-%d')
    resources = TextAreaField('Resources Required', validators=[Optional(), Length(max=500)])
    expected_outcomes = TextAreaField('Expected Outcomes', validators=[DataRequired(), Length(min=10, max=500)])
    status = SelectField('Status', validators=[Optional()], choices=[
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled')
    ], default='pending')
    submit = SubmitField('Save Intervention')

    def validate_end_date(self, end_date):
        """Validate that end date is after start date"""
        if end_date.data and self.start_date.data:
            if end_date.data < self.start_date.data:
                raise ValidationError('End date must be after start date')

    def validate_follow_up_date(self, follow_up_date):
        """Validate that follow-up date is not before start date"""
        if follow_up_date.data and self.start_date.data:
            if follow_up_date.data < self.start_date.data:
                raise ValidationError('Follow-up date cannot be before start date')

class FeedbackForm(FlaskForm):
    """Form for providing feedback on interventions"""
    comments = TextAreaField('Comments', validators=[DataRequired(), Length(min=10, max=1000)])
    effectiveness = SelectField('Effectiveness', validators=[DataRequired()], choices=[
        ('1', 'Not Effective'),
        ('2', 'Slightly Effective'),
        ('3', 'Moderately Effective'),
        ('4', 'Very Effective'),
        ('5', 'Extremely Effective')
    ])
    follow_up_needed = BooleanField('Additional Follow-up Needed')
    submit = SubmitField('Submit Feedback')

class StudentDataForm(FlaskForm):
    """Form for updating student academic data"""
    units_enrolled = IntegerField('Units Enrolled', validators=[DataRequired(), NumberRange(min=0)])
    units_approved = IntegerField('Units Approved', validators=[DataRequired(), NumberRange(min=0)])
    current_grade = FloatField('Current Grade', validators=[DataRequired(), NumberRange(min=0, max=20)])
    attendance_rate = FloatField('Attendance Rate (%)', validators=[DataRequired(), NumberRange(min=0, max=100)])
    submit = SubmitField('Update & Recalculate')

    def validate_units_approved(self, units_approved):
        """Validate that approved units do not exceed enrolled units"""
        if units_approved.data > self.units_enrolled.data:
            raise ValidationError('Approved units cannot exceed enrolled units')

class PredictionRequestForm(FlaskForm):
    """Form for requesting a new prediction"""
    student_id = SelectField('Student', coerce=int, validators=[DataRequired()])
    include_latest_data = BooleanField('Include Latest Academic Data', default=True)
    submit = SubmitField('Generate Prediction')

class CourseForm(FlaskForm):
    """Form for creating/editing courses"""
    name = StringField('Course Name', validators=[DataRequired(), Length(min=3, max=100)])
    code = StringField('Course Code', validators=[DataRequired(), Length(min=2, max=20)])
    department_id = SelectField('Department', coerce=int, validators=[DataRequired()])
    description = TextAreaField('Description', validators=[Optional(), Length(max=1000)])
    credits = IntegerField('Credits', validators=[DataRequired(), NumberRange(min=1)])
    max_students = IntegerField('Maximum Students', validators=[DataRequired(), NumberRange(min=1)])
    active = BooleanField('Active', default=True)
    submit = SubmitField('Save Course')

class DepartmentForm(FlaskForm):
    """Form for creating/editing departments"""
    name = StringField('Department Name', validators=[DataRequired(), Length(min=3, max=100)])
    code = StringField('Department Code', validators=[DataRequired(), Length(min=2, max=20)])
    faculty_head_id = SelectField('Faculty Head', coerce=int, validators=[Optional()])
    description = TextAreaField('Description', validators=[Optional(), Length(max=1000)])
    submit = SubmitField('Save Department')

class SystemSettingsForm(FlaskForm):
    """Form for editing system settings"""
    site_name = StringField('Site Name', validators=[DataRequired(), Length(min=3, max=100)])
    contact_email = StringField('Contact Email', validators=[DataRequired(), Email()])
    items_per_page = IntegerField('Items Per Page', validators=[DataRequired(), NumberRange(min=5, max=100)])
    
    # Prediction model settings
    model_type = SelectField('Model Type', validators=[DataRequired()], choices=[
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('xgboost', 'XGBoost'),
        ('neural_network', 'Neural Network')
    ])
    prediction_threshold = FloatField('Prediction Threshold', validators=[
        DataRequired(), 
        NumberRange(min=0.1, max=0.9, message='Threshold must be between 0.1 and 0.9')
    ])
    auto_update_predictions = BooleanField('Auto-update Predictions')
    auto_update_frequency = SelectField('Auto-update Frequency', validators=[Optional()], choices=[
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly')
    ])
    
    # Notification settings
    email_notifications = BooleanField('Email Notifications')
    high_risk_alerts = BooleanField('High Risk Alerts')
    intervention_reminders = BooleanField('Intervention Reminders')
    
    submit = SubmitField('Save Settings')

class ExportDataForm(FlaskForm):
    """Form for exporting data"""
    data_type = SelectField('Data to Export', validators=[DataRequired()], choices=[
        ('at-risk-students', 'At-Risk Students'),
        ('all-students', 'All Students'),
        ('course-performance', 'Course Performance Data'),
        ('interventions', 'Intervention Records'),
        ('analytics', 'Complete Analytics Report')
    ])
    format = SelectField('Format', validators=[DataRequired()], choices=[
        ('csv', 'CSV'),
        ('excel', 'Excel'),
        ('pdf', 'PDF Report')
    ])
    time_range = SelectField('Time Range', validators=[DataRequired()], choices=[
        ('current', 'Current Semester'),
        ('year', 'Academic Year'),
        ('all', 'All Available Data'),
        ('custom', 'Custom Range')
    ])
    start_date = DateField('Start Date', validators=[Optional()], format='%Y-%m-%d')
    end_date = DateField('End Date', validators=[Optional()], format='%Y-%m-%d')
    include_analysis = BooleanField('Include Analysis & Recommendations')
    submit = SubmitField('Export Data')

    def validate_end_date(self, end_date):
        """Validate that end date is after start date"""
        if end_date.data and self.start_date.data:
            if end_date.data < self.start_date.data:
                raise ValidationError('End date must be after start date')
        
    def validate_start_date(self, start_date):
        """Validate that start date is provided when custom range is selected"""
        if self.time_range.data == 'custom' and not start_date.data:
            raise ValidationError('Start date is required for custom range')
    
    def validate_end_date(self, end_date):
        """Validate that end date is provided when custom range is selected"""
        if self.time_range.data == 'custom' and not end_date.data:
            raise ValidationError('End date is required for custom range')

class UserForm(FlaskForm):
    """Form for creating/editing users"""
    first_name = StringField('First Name', validators=[DataRequired(), Length(min=2, max=50)])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    role = SelectField('Role', validators=[DataRequired()], choices=[
        ('student', 'Student'),
        ('faculty', 'Faculty'),
        ('admin', 'Administrator')
    ])
    active = BooleanField('Active', default=True)
    
    # Student-specific fields
    student_id = StringField('Student ID', validators=[Optional(), Length(max=20)])
    course_id = SelectField('Course', validators=[Optional()], coerce=int)
    
    # Faculty-specific fields
    faculty_id = StringField('Faculty ID', validators=[Optional(), Length(max=20)])
    department_id = SelectField('Department', validators=[Optional()], coerce=int)
    
    # For new users only
    password = PasswordField('Password', validators=[Optional(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[
        Optional(),
        EqualTo('password', message='Passwords must match')
    ])
    
    submit = SubmitField('Save User')
    
    def validate(self, extra_validators=None):
        """Custom validation based on role"""
        if not super().validate(extra_validators):
            return False
        
        if self.role.data == 'student':
            if not self.student_id.data:
                self.student_id.errors.append('Student ID is required for student role')
                return False
            if not self.course_id.data:
                self.course_id.errors.append('Course is required for student role')
                return False
        elif self.role.data == 'faculty':
            if not self.faculty_id.data:
                self.faculty_id.errors.append('Faculty ID is required for faculty role')
                return False
            if not self.department_id.data:
                self.department_id.errors.append('Department is required for faculty role')
                return False
        
        return True

class ModelRetrainingForm(FlaskForm):
    """Form for retraining the prediction model"""
    model_type = SelectField('Model Type', validators=[DataRequired()], choices=[
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('xgboost', 'XGBoost'),
        ('neural_network', 'Neural Network')
    ])
    dataset = SelectField('Training Dataset', validators=[DataRequired()], choices=[
        ('current', 'Current Academic Year'),
        ('full', 'Full Historical Data'),
        ('custom', 'Custom Dataset')
    ])
    hyperparameter_tuning = SelectField('Hyperparameter Tuning', validators=[DataRequired()], choices=[
        ('default', 'Use Default Parameters'),
        ('grid', 'Grid Search'),
        ('random', 'Random Search')
    ])
    feature_selection = BooleanField('Perform Automatic Feature Selection')
    cross_validation = BooleanField('Use Cross-validation', default=True)
    submit = SubmitField('Start Retraining')

class BackupForm(FlaskForm):
    """Form for backing up the system"""
    backup_type = SelectField('Backup Type', validators=[DataRequired()], choices=[
        ('full', 'Full System Backup'),
        ('data', 'Data Only'),
        ('config', 'Configuration Only'),
        ('models', 'ML Models Only')
    ])
    location = SelectField('Backup Location', validators=[DataRequired()], choices=[
        ('local', 'Local Server'),
        ('cloud', 'Cloud Storage'),
        ('download', 'Download to Computer')
    ])
    encrypt = BooleanField('Encrypt Backup', default=True)
    schedule = BooleanField('Schedule Regular Backups')
    frequency = SelectField('Backup Frequency', validators=[Optional()], choices=[
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly')
    ])
    submit = SubmitField('Start Backup')