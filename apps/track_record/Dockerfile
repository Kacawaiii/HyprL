FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements_dashboard.txt .
RUN pip install --no-cache-dir -r requirements_dashboard.txt

# Copy App Code
# We assume the build context is the root of the repo to capture 'apps' and 'requirements'
# But for a simpler standalone build, we copy specific folders.
# Adjust this based on where you run 'docker build' from.
# Here we assume context is 'apps/track_record' or we copy relatively if context is root.

# If context is ROOT (recommended):
COPY apps/track_record /app/apps/track_record
# We need to mimic the path structure for imports to work if they use 'apps.track_record...'
# The app adds project root to path usually, or we run as module.

# Set PYTHONPATH to include /app so "apps.track_record" imports work
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose Streamlit Port
EXPOSE 8501

# Run
CMD ["streamlit", "run", "apps/track_record/streamlit_app.py", "--server.address=0.0.0.0"]
