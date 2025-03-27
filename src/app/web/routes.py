from flask import render_template, redirect, url_for, flash, request, jsonify, abort, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime

from src.app.web import web
from src.app.models.user import User
from src.app.models.student import Student
from src.app.models.course import Course
from src.app.models.department import Department
from src.app.models.prediction import Prediction
from src.app.models.intervention import Intervention
from src.app.utils.decorators import role_required
from src.app.web.forms import ContactForm, UserProfileForm, InterventionForm

# Mock data for development - Replace with database queries in production
from src.app.utils.mock_data import get_mock_data

# =====================
# Public Routes
# =====================

@web.route('/')
def index():
    """Home/landing page"""
    return render_template('index.html')

@web.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@web.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact page with form submission"""
    form = ContactForm()
    if form.validate_on_submit():
        # Process form data (e.g., send email)
        flash('Your message has been sent! We will get back to you soon.', 'success')
        return redirect(url_for('web.contact'))
    return render_template('contact.html', form=form)

@web.route('/privacy')
def privacy():
    """Privacy policy page"""
    return render_template('legal/privacy.html')

@web.route('/terms')
def terms():
    """Terms of service page"""
    return render_template('legal/terms.html')

# =====================
# Dashboard Routes
# =====================

@web.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard - redirects to appropriate role-based dashboard"""
    if current_user.role == 'student':
        return redirect(url_for('web.student_dashboard'))
    elif current_user.role == 'faculty':
        return redirect(url_for('web.faculty_dashboard'))
    elif current_user.role == 'admin':
        return redirect(url_for('web.admin_dashboard'))
    else:
        flash('Invalid user role', 'danger')
        return redirect(url_for('web.index'))

@web.route('/dashboard/student')
@login_required
@role_required('student')
def student_dashboard():
    """Student dashboard with academic status and predictions"""
    # Get student data
    try:
        # In production, retrieve from database:
        # student = Student.query.filter_by(user_id=current_user.id).first_or_404()
        # prediction = Prediction.get_latest_for_student(student.id)
        
        # Development mock data
        student_data = get_mock_data('student')
        
        # Get risk factors
        risk_factors = get_mock_data('risk_factors')
        risk_factors_detailed = get_mock_data('risk_factors_detailed')
        
        # Get improvement recommendations
        improvements = get_mock_data('improvements')
        
        # Get academic record
        academic_record = get_mock_data('academic_record')
        
        # Determine status and risk colors
        status_mapping = {
            'Excellent': 'success',
            'Good': 'info',
            'Average': 'warning',
            'At Risk': 'danger'
        }
        
        risk_mapping = {
            'Low': 'success',
            'Medium': 'warning',
            'High': 'danger'
        }
        
        status_color = status_mapping.get(student_data['status'], 'info')
        risk_color = risk_mapping.get(student_data['risk_level'], 'info')
        
        # Get performance data for charts
        semester1_grades = [15, 14, 13, 16, 17, 15]
        semester2_grades = [14, 15, 16, 15, 14, 13]
        semester1_units = [6, 5, 6, 6, 6, 5]
        semester2_units = [5, 6, 6, 5, 6, 6]
        semester1_risk = [20, 25, 15, 10, 15, 20]
        semester2_risk = [15, 10, 10, 15, 20, 25]
        
        # Calculate class average for comparison
        class_average = 70
        performance_comparison = student_data['overall_percentage'] - class_average
        comparison_color = 'success' if performance_comparison > 0 else 'danger'
        
        return render_template(
            'dashboard/student_dashboard.html',
            student=student_data,
            status_color=status_color,
            risk_color=risk_color,
            risk_factors=risk_factors,
            risk_factors_detailed=risk_factors_detailed,
            improvements=improvements,
            academic_record=academic_record,
            semester1_grades=semester1_grades,
            semester2_grades=semester2_grades,
            semester1_units=semester1_units,
            semester2_units=semester2_units,
            semester1_risk=semester1_risk,
            semester2_risk=semester2_risk,
            class_average=class_average,
            performance_comparison=performance_comparison,
            comparison_color=comparison_color
        )
    except Exception as e:
        current_app.logger.error(f"Error in student dashboard: {str(e)}")
        flash('An error occurred while loading your dashboard. Please try again.', 'danger')
        return redirect(url_for('web.index'))

@web.route('/dashboard/faculty')
@login_required
@role_required('faculty')
def faculty_dashboard():
    """Faculty dashboard with student monitoring and analytics"""
    try:
        # In production, retrieve from database
        # faculty = Faculty.query.filter_by(user_id=current_user.id).first_or_404()
        
        # Development mock data
        stats = get_mock_data('faculty_stats')
        at_risk_students = get_mock_data('at_risk_students')
        risk_factors = get_mock_data('faculty_risk_factors')
        recent_interventions = get_mock_data('recent_interventions')
        courses = get_mock_data('faculty_courses')
        
        # Data for charts
        course_names = json.dumps([course['name'] for course in courses])
        course_grades = json.dumps([course['avg_grade'] for course in courses])
        course_completion = json.dumps([course['completion_rate'] for course in courses])
        course_risk = json.dumps([course['at_risk_percentage'] for course in courses])
        
        trend_months = json.dumps(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
        trend_grades = json.dumps([14.2, 14.5, 14.8, 15.0, 14.7, 14.9])
        trend_completion = json.dumps([75, 78, 80, 82, 80, 83])
        trend_risk = json.dumps([18, 15, 12, 10, 13, 11])
        
        return render_template(
            'dashboard/faculty_dashboard.html',
            stats=stats,
            at_risk_students=at_risk_students,
            risk_factors=risk_factors,
            recent_interventions=recent_interventions,
            courses=courses,
            course_names=course_names,
            course_grades=course_grades,
            course_completion=course_completion,
            course_risk=course_risk,
            trend_months=trend_months,
            trend_grades=trend_grades,
            trend_completion=trend_completion,
            trend_risk=trend_risk
        )
    except Exception as e:
        current_app.logger.error(f"Error in faculty dashboard: {str(e)}")
        flash('An error occurred while loading your dashboard. Please try again.', 'danger')
        return redirect(url_for('web.index'))

@web.route('/dashboard/admin')
@login_required
@role_required('admin')
def admin_dashboard():
    """Admin dashboard with system metrics and management"""
    try:
        # In production, retrieve from database
        
        # Development mock data
        system_stats = get_mock_data('system_stats')
        system_logs = get_mock_data('system_logs')
        system_resources = get_mock_data('system_resources')
        departments = get_mock_data('departments')
        users = get_mock_data('users')
        model_info = get_mock_data('model_info')
        model_metrics = get_mock_data('model_metrics')
        
        # Data for charts
        months = json.dumps(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
        
        performance_data = {
            'grades': json.dumps([14.2, 14.5, 14.8, 15.0, 14.7, 14.9]),
            'completion': json.dumps([75, 78, 80, 82, 80, 83]),
            'risk': json.dumps([18, 15, 12, 10, 13, 11]),
            'retention': json.dumps([85, 87, 90, 92, 91, 93])
        }
        
        user_activity = {
            'dates': json.dumps(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
            'counts': json.dumps([120, 132, 145, 150, 143, 85, 75]),
            'faculty_counts': json.dumps([25, 30, 28, 32, 30, 15, 10]),
            'student_counts': json.dumps([90, 95, 110, 112, 108, 65, 60]),
            'admin_counts': json.dumps([5, 7, 7, 6, 5, 5, 5])
        }
        
        user_roles = {
            'faculty': 45,
            'students': 350,
            'admins': 8
        }
        
        resource_history = {
            'timestamps': json.dumps(['8:00', '9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00']),
            'cpu': json.dumps([25, 30, 40, 60, 45, 35, 30, 25]),
            'memory': json.dumps([40, 42, 45, 50, 55, 58, 60, 58]),
            'disk': json.dumps([62, 62, 63, 63, 64, 64, 65, 65]),
            'db': json.dumps([35, 40, 45, 50, 55, 45, 40, 35])
        }
        
        top_user_actions = get_mock_data('top_user_actions')
        
        # Model visualization data
        confusion_matrix = json.dumps([
            {'x': 'Dropout', 'y': 'Dropout', 'v': 87},
            {'x': 'Dropout', 'y': 'Graduate', 'v': 12},
            {'x': 'Dropout', 'y': 'Enrolled', 'v': 8},
            {'x': 'Graduate', 'y': 'Dropout', 'v': 5},
            {'x': 'Graduate', 'y': 'Graduate', 'v': 92},
            {'x': 'Graduate', 'y': 'Enrolled', 'v': 3},
            {'x': 'Enrolled', 'y': 'Dropout', 'v': 8},
            {'x': 'Enrolled', 'y': 'Graduate', 'v': 6},
            {'x': 'Enrolled', 'y': 'Enrolled', 'v': 89}
        ])
        
        feature_importance = {
            'features': json.dumps([
                '1st sem grade', '2nd sem grade', 'Attendance', 'Age',
                'Previous qualifications', 'Financial status', 'Debtor status', 
                'Study time', 'Curricular units'
            ]),
            'importance': json.dumps([0.23, 0.21, 0.15, 0.12, 0.09, 0.08, 0.06, 0.04, 0.02])
        }
        
        return render_template(
            'admin/admin_dashboard.html',
            system_stats=system_stats,
            system_logs=system_logs,
            system_resources=system_resources,
            departments=departments,
            users=users,
            model_info=model_info,
            model_metrics=model_metrics,
            months=months,
            performance_data=performance_data,
            user_activity=user_activity,
            user_roles=user_roles,
            resource_history=resource_history,
            top_user_actions=top_user_actions,
            confusion_matrix=confusion_matrix,
            feature_importance=feature_importance
        )
    except Exception as e:
        current_app.logger.error(f"Error in admin dashboard: {str(e)}")
        flash('An error occurred while loading the admin dashboard. Please try again.', 'danger')
        return redirect(url_for('web.index'))

# =====================
# Student Routes
# =====================

@web.route('/students')
@login_required
@role_required(['faculty', 'admin'])
def student_list():
    """List of all students (for faculty and admin)"""
    # Get filter parameters
    course_id = request.args.get('course', type=int)
    risk_level = request.args.get('risk')
    search_query = request.args.get('q')
    
    # In production, query database with filters
    # students = Student.query
    # if course_id:
    #     students = students.filter_by(course_id=course_id)
    # if risk_level:
    #     students = students.filter_by(risk_level=risk_level)
    # if search_query:
    #     students = students.filter(Student.name.ilike(f'%{search_query}%'))
    # students = students.all()
    
    # Development mock data
    students = get_mock_data('all_students')
    
    # Filter mock data based on parameters
    if course_id:
        students = [s for s in students if s['course_id'] == course_id]
    if risk_level:
        students = [s for s in students if s['risk_level'].lower() == risk_level.lower()]
    if search_query:
        students = [s for s in students if search_query.lower() in s['name'].lower()]
    
    # Get courses for filter dropdown
    courses = get_mock_data('courses')
    
    return render_template(
        'students/student_list.html',
        students=students,
        courses=courses,
        selected_course=course_id,
        selected_risk=risk_level,
        search_query=search_query
    )

@web.route('/students/<int:id>')
@login_required
def student_detail(id):
    """Student detail page"""
    # Check permissions - students can only view their own profile
    if current_user.role == 'student' and current_user.id != id and not current_user.is_admin():
        flash('You do not have permission to view this student\'s details.', 'danger')
        return redirect(url_for('web.dashboard'))
    
    # In production, query database
    # student = Student.query.get_or_404(id)
    # predictions = Prediction.query.filter_by(student_id=id).order_by(Prediction.created_at.desc()).all()
    # interventions = Intervention.query.filter_by(student_id=id).order_by(Intervention.created_at.desc()).all()
    
    # Development mock data
    student = next((s for s in get_mock_data('all_students') if s['id'] == id), None)
    if not student:
        abort(404)
    
    predictions = get_mock_data('student_predictions')
    interventions = get_mock_data('student_interventions')
    
    # Prepare data for charts
    prediction_dates = json.dumps([p['date'] for p in predictions])
    prediction_risks = json.dumps([p['risk_score'] for p in predictions])
    academic_data = get_mock_data('student_academic_data')
    
    return render_template(
        'students/student_detail.html',
        student=student,
        predictions=predictions,
        interventions=interventions,
        prediction_dates=prediction_dates,
        prediction_risks=prediction_risks,
        academic_data=academic_data
    )

@web.route('/recommendations')
@login_required
@role_required('student')
def recommendations():
    """Detailed recommendations for students"""
    # In production, query database
    # student = Student.query.filter_by(user_id=current_user.id).first_or_404()
    # recommendations = Recommendation.query.filter_by(student_id=student.id).all()
    
    # Development mock data
    detailed_recommendations = get_mock_data('detailed_recommendations')
    
    return render_template(
        'students/recommendations.html',
        recommendations=detailed_recommendations
    )

# =====================
# Faculty Routes
# =====================

@web.route('/interventions')
@login_required
@role_required(['faculty', 'admin'])
def interventions():
    """List of all interventions"""
    # Get filter parameters
    status = request.args.get('status')
    student_id = request.args.get('student_id', type=int)
    
    # In production, query database with filters
    # interventions = Intervention.query
    # if status:
    #     interventions = interventions.filter_by(status=status)
    # if student_id:
    #     interventions = interventions.filter_by(student_id=student_id)
    # interventions = interventions.order_by(Intervention.created_at.desc()).all()
    
    # Development mock data
    interventions_data = get_mock_data('all_interventions')
    
    # Filter mock data based on parameters
    if status:
        interventions_data = [i for i in interventions_data if i['status'].lower() == status.lower()]
    if student_id:
        interventions_data = [i for i in interventions_data if i['student_id'] == student_id]
    
    return render_template(
        'faculty/interventions.html',
        interventions=interventions_data
    )

@web.route('/interventions/new', methods=['GET', 'POST'])
@login_required
@role_required(['faculty', 'admin'])
def new_intervention():
    """Create a new intervention"""
    form = InterventionForm()
    
    # Populate student dropdown
    form.student_id.choices = [(s['id'], s['name']) for s in get_mock_data('all_students')]
    
    if form.validate_on_submit():
        # In production, save to database
        # intervention = Intervention(
        #     student_id=form.student_id.data,
        #     faculty_id=current_user.id,
        #     type=form.type.data,
        #     description=form.description.data,
        #     start_date=form.start_date.data,
        #     end_date=form.end_date.data,
        #     follow_up_date=form.follow_up_date.data,
        #     status='Pending'
        # )
        # db.session.add(intervention)
        # db.session.commit()
        
        flash('Intervention has been created successfully!', 'success')
        return redirect(url_for('web.interventions'))
    
    return render_template(
        'faculty/intervention_form.html',
        form=form,
        title='New Intervention'
    )

@web.route('/interventions/<int:id>', methods=['GET', 'POST'])
@login_required
@role_required(['faculty', 'admin'])
def edit_intervention(id):
    """Edit an existing intervention"""
    # In production, query database
    # intervention = Intervention.query.get_or_404(id)
    
    # Development mock data
    intervention = next((i for i in get_mock_data('all_interventions') if i['id'] == id), None)
    if not intervention:
        abort(404)
    
    form = InterventionForm(obj=intervention)
    
    # Populate student dropdown
    form.student_id.choices = [(s['id'], s['name']) for s in get_mock_data('all_students')]
    
    if form.validate_on_submit():
        # In production, update database
        # intervention.type = form.type.data
        # intervention.description = form.description.data
        # intervention.start_date = form.start_date.data
        # intervention.end_date = form.end_date.data
        # intervention.follow_up_date = form.follow_up_date.data
        # intervention.status = form.status.data
        # db.session.commit()
        
        flash('Intervention has been updated successfully!', 'success')
        return redirect(url_for('web.interventions'))
    
    return render_template(
        'faculty/intervention_form.html',
        form=form,
        intervention=intervention,
        title='Edit Intervention'
    )

@web.route('/course/<int:id>')
@login_required
def course_detail(id):
    """Course detail page"""
    # In production, query database
    # course = Course.query.get_or_404(id)
    
    # Development mock data
    course = next((c for c in get_mock_data('courses') if c['id'] == id), None)
    if not course:
        abort(404)
    
    # Get students in this course
    students = [s for s in get_mock_data('all_students') if s['course_id'] == id]
    
    return render_template(
        'faculty/course_detail.html',
        course=course,
        students=students
    )

@web.route('/course/<int:id>/students')
@login_required
@role_required(['faculty', 'admin'])
def course_students(id):
    """Students in a specific course"""
    # In production, query database
    # course = Course.query.get_or_404(id)
    # students = Student.query.filter_by(course_id=id).all()
    
    # Development mock data
    course = next((c for c in get_mock_data('courses') if c['id'] == id), None)
    if not course:
        abort(404)
    
    students = [s for s in get_mock_data('all_students') if s['course_id'] == id]
    
    return render_template(
        'faculty/course_students.html',
        course=course,
        students=students
    )

@web.route('/analytics')
@login_required
@role_required(['faculty', 'admin'])
def analytics():
    """Advanced analytics page"""
    # Development mock data
    analytics_data = get_mock_data('analytics_data')
    
    return render_template(
        'faculty/analytics.html',
        analytics_data=analytics_data
    )

# =====================
# Admin Routes
# =====================

@web.route('/user-management')
@login_required
@role_required('admin')
def user_management():
    """User management page for admins"""
    # In production, query database
    # users = User.query.all()
    
    # Development mock data
    users = get_mock_data('all_users')
    
    return render_template(
        'admin/user_management.html',
        users=users
    )

@web.route('/system-settings')
@login_required
@role_required('admin')
def system_settings():
    """System settings page for admins"""
    # Development mock data
    settings = get_mock_data('system_settings')
    
    return render_template(
        'admin/system_settings.html',
        settings=settings
    )

@web.route('/system-logs')
@login_required
@role_required('admin')
def system_logs():
    """System logs page for admins"""
    # Development mock data
    logs = get_mock_data('all_system_logs')
    
    return render_template(
        'admin/system_logs.html',
        logs=logs
    )

@web.route('/department/<int:id>')
@login_required
@role_required(['faculty', 'admin'])
def department_detail(id):
    """Department detail page"""
    # In production, query database
    # department = Department.query.get_or_404(id)
    
    # Development mock data
    department = next((d for d in get_mock_data('departments') if d['id'] == id), None)
    if not department:
        abort(404)
    
    return render_template(
        'admin/department_detail.html',
        department=department
    )

@web.route('/data-management')
@login_required
@role_required('admin')
def data_management():
    """Data management page for admins"""
    return render_template('admin/data_management.html')

@web.route('/reports')
@login_required
@role_required(['faculty', 'admin'])
def reports():
    """Generate reports page"""
    return render_template('admin/reports.html')

# =====================
# User Profile Routes
# =====================

@web.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('profile/profile.html')

@web.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile"""
    # In production, get user data
    # user = User.query.get(current_user.id)
    
    form = UserProfileForm()
    
    if request.method == 'GET':
        # Pre-populate form with user data
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email
    
    if form.validate_on_submit():
        # In production, update user data
        # user.first_name = form.first_name.data
        # user.last_name = form.last_name.data
        # user.email = form.email.data
        # db.session.commit()
        
        flash('Your profile has been updated successfully!', 'success')
        return redirect(url_for('web.profile'))
    
    return render_template('profile/edit_profile.html', form=form)

@web.route('/settings')
@login_required
def settings():
    """User settings page"""
    return render_template('profile/settings.html')

# =====================
# Support Routes
# =====================

@web.route('/support')
def support():
    """Support page"""
    return render_template('support/support.html')

@web.route('/documentation')
def documentation():
    """Documentation page"""
    return render_template('support/documentation.html')

# =====================
# API Routes for AJAX
# =====================

@web.route('/api/student/<int:id>/data')
@login_required
def student_data_api(id):
    """API endpoint to get student data for AJAX requests"""
    # Check permissions
    if current_user.role == 'student' and current_user.id != id and not current_user.is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    # In production, query database
    # student = Student.query.get_or_404(id)
    # data = student.to_dict()
    
    # Development mock data
    student = next((s for s in get_mock_data('all_students') if s['id'] == id), None)
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    
    return jsonify(student)

@web.route('/api/refresh-predictions', methods=['POST'])
@login_required
@role_required(['faculty', 'admin'])
def refresh_predictions():
    """API endpoint to trigger prediction refresh"""
    # In production, this would trigger the machine learning pipeline
    # to generate new predictions
    
    # Simulate processing time
    import time
    time.sleep(2)
    
    return jsonify({'status': 'success', 'message': 'Predictions refreshed successfully'})

# =====================
# Error handlers
# =====================

@web.app_errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('errors/404.html'), 404

@web.app_errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('errors/500.html'), 500

@web.app_errorhandler(403)
def forbidden(e):
    """Handle 403 errors"""
    return render_template('errors/403.html'), 403