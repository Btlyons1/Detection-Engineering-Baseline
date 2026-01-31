"""
Synthetic CloudTrail Data Generator for Detection Baselining
============================================================

Generates realistic CloudTrail-like data with intentional anomalies
for demonstrating detection baseline techniques.

Author: Security Engineering Team
Version: 1.1.0
"""

import json
import random
import duckdb
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Configuration
SEED = 42
random.seed(SEED)

# Define realistic user patterns
USERS = {
    "alice.smith": {"role": "developer", "typical_hours": (8, 18), "api_frequency": "high"},
    "bob.jones": {"role": "developer", "typical_hours": (9, 17), "api_frequency": "medium"},
    "carol.williams": {"role": "data_engineer", "typical_hours": (7, 19), "api_frequency": "very_high"},
    "david.brown": {"role": "sre", "typical_hours": (0, 24), "api_frequency": "high"},  # On-call
    "eve.davis": {"role": "security", "typical_hours": (9, 18), "api_frequency": "low"},
    "service-account-ci": {"role": "service", "typical_hours": (0, 24), "api_frequency": "very_high"},
    "service-account-backup": {"role": "service", "typical_hours": (2, 4), "api_frequency": "medium"},
}

# API call patterns by role
API_PATTERNS = {
    "developer": {
        "common": ["ec2:DescribeInstances", "s3:GetObject", "s3:PutObject", "logs:GetLogEvents",
                   "cloudwatch:GetMetricData", "iam:GetUser", "sts:GetCallerIdentity"],
        "occasional": ["ec2:RunInstances", "ec2:TerminateInstances", "lambda:InvokeFunction"],
        "rare": ["iam:CreateUser", "iam:AttachUserPolicy"]
    },
    "data_engineer": {
        "common": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "glue:GetTable",
                   "athena:StartQueryExecution", "redshift:ExecuteStatement"],
        "occasional": ["glue:CreateJob", "glue:StartJobRun", "emr:RunJobFlow"],
        "rare": ["iam:PassRole", "kms:Decrypt"]
    },
    "sre": {
        "common": ["ec2:DescribeInstances", "cloudwatch:GetMetricData", "logs:FilterLogEvents",
                   "ecs:DescribeServices", "elasticloadbalancing:DescribeTargetHealth"],
        "occasional": ["ec2:StopInstances", "ec2:StartInstances", "ecs:UpdateService"],
        "rare": ["iam:UpdateAssumeRolePolicy", "secretsmanager:GetSecretValue"]
    },
    "security": {
        "common": ["cloudtrail:LookupEvents", "guardduty:ListFindings", "securityhub:GetFindings",
                   "iam:ListUsers", "iam:ListRoles"],
        "occasional": ["iam:GetPolicy", "iam:GetRolePolicy", "config:GetComplianceDetailsByResource"],
        "rare": ["iam:CreatePolicy", "organizations:DescribeOrganization"]
    },
    "service": {
        "common": ["s3:GetObject", "s3:PutObject", "dynamodb:GetItem", "dynamodb:PutItem",
                   "sqs:SendMessage", "sqs:ReceiveMessage", "sns:Publish"],
        "occasional": ["lambda:InvokeFunction", "secretsmanager:GetSecretValue"],
        "rare": []
    }
}

# Suspicious patterns to inject
SUSPICIOUS_PATTERNS = [
    {"user": "alice.smith", "api": "iam:CreateAccessKey", "hour": 3, "description": "Off-hours credential creation"},
    {"user": "bob.jones", "api": "s3:GetObject", "count": 500, "description": "Data exfiltration attempt"},
    {"user": "unknown-user-ext", "api": "sts:AssumeRole", "description": "Unknown external user"},
    {"user": "carol.williams", "api": "kms:Decrypt", "count": 100, "description": "Unusual KMS activity"},
]

SOURCE_IPS = [
    "10.0.1.50", "10.0.1.51", "10.0.1.52", "10.0.2.100", "10.0.2.101",
    "192.168.1.10", "192.168.1.11", "172.16.0.5", "172.16.0.6"
]

EXTERNAL_IPS = ["203.0.113.50", "198.51.100.25", "192.0.2.100"]

REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]

def generate_event_id():
    """Generate a realistic CloudTrail event ID."""
    return hashlib.sha256(str(random.random()).encode()).hexdigest()[:36]

def generate_request_id():
    """Generate a realistic AWS request ID."""
    return f"{random.randint(10000000, 99999999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(100000000000, 999999999999)}"

def select_api_call(role: str) -> str:
    """Select an API call based on role with weighted probability."""
    patterns = API_PATTERNS.get(role, API_PATTERNS["developer"])

    r = random.random()
    if r < 0.85:  # 85% common
        return random.choice(patterns["common"])
    elif r < 0.97:  # 12% occasional
        return random.choice(patterns["occasional"]) if patterns["occasional"] else random.choice(patterns["common"])
    else:  # 3% rare
        return random.choice(patterns["rare"]) if patterns["rare"] else random.choice(patterns["occasional"] or patterns["common"])

def is_within_hours(hour: int, typical_hours: tuple) -> bool:
    """Check if hour is within typical working hours."""
    start, end = typical_hours
    if start <= end:
        return start <= hour < end
    else:  # Handles overnight shifts
        return hour >= start or hour < end

def generate_event(user: str, user_info: dict, timestamp: datetime, is_anomaly: bool = False,
                   anomaly_api: str = None, source_ip: str = None) -> dict:
    """Generate a single CloudTrail-like event."""

    if is_anomaly and anomaly_api:
        api_call = anomaly_api
    else:
        api_call = select_api_call(user_info["role"])

    service, action = api_call.split(":")

    # Determine source IP
    if source_ip:
        ip = source_ip
    elif user.startswith("service-account"):
        ip = random.choice(SOURCE_IPS[:5])  # Internal IPs for service accounts
    elif "unknown" in user:
        ip = random.choice(EXTERNAL_IPS)
    else:
        ip = random.choice(SOURCE_IPS)

    # Error rate varies by API type
    error_rate = 0.02 if "Describe" in action or "Get" in action or "List" in action else 0.05
    is_error = random.random() < error_rate

    event = {
        "eventID": generate_event_id(),
        "eventTime": timestamp.isoformat() + "Z",
        "eventSource": f"{service}.amazonaws.com",
        "eventName": action,
        "awsRegion": random.choice(REGIONS),
        "sourceIPAddress": ip,
        "userAgent": "aws-sdk-python/1.26.0" if user.startswith("service") else "console.aws.amazon.com",
        "userIdentity": {
            "type": "IAMUser" if not user.startswith("service") else "AssumedRole",
            "userName": user,
            "arn": f"arn:aws:iam::123456789012:user/{user}",
            "accountId": "123456789012"
        },
        "requestID": generate_request_id(),
        "errorCode": "AccessDenied" if is_error else None,
        "errorMessage": "User is not authorized" if is_error else None,
        "requestParameters": {},
        "responseElements": {} if not is_error else None,
        "readOnly": "Describe" in action or "Get" in action or "List" in action,
        "eventType": "AwsApiCall",
        "managementEvent": True,
        "eventCategory": "Management"
    }

    return event

def generate_dataset(days: int = 30, events_per_day_base: int = 1000) -> list:
    """Generate a full dataset with normal and anomalous patterns."""

    events = []
    start_date = datetime.now() - timedelta(days=days)

    print(f"Generating {days} days of CloudTrail data...")

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)

        # Weekend adjustment - less activity
        is_weekend = current_date.weekday() >= 5
        day_multiplier = 0.3 if is_weekend else 1.0

        events_today = int(events_per_day_base * day_multiplier * random.uniform(0.8, 1.2))

        for _ in range(events_today):
            # Select user with weighted probability
            user = random.choices(
                list(USERS.keys()),
                weights=[3, 2, 4, 2, 1, 4, 1],  # Service accounts active but below outlier threshold
                k=1
            )[0]
            user_info = USERS[user]

            # Generate timestamp based on typical hours
            hour = random.randint(0, 23)
            if not is_within_hours(hour, user_info["typical_hours"]):
                if random.random() > 0.05:  # 95% of events during normal hours
                    start_h, end_h = user_info["typical_hours"]
                    if start_h <= end_h:
                        hour = random.randint(start_h, end_h - 1)
                    else:
                        hour = random.randint(start_h, 23) if random.random() > 0.5 else random.randint(0, end_h - 1)

            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            timestamp = current_date.replace(hour=hour, minute=minute, second=second)

            event = generate_event(user, user_info, timestamp)
            events.append(event)

        # Inject anomalies on specific days
        if day_offset == 15:  # Mid-month anomaly: off-hours credential creation
            anomaly_time = current_date.replace(hour=3, minute=15, second=30)
            event = generate_event(
                "alice.smith", USERS["alice.smith"], anomaly_time,
                is_anomaly=True, anomaly_api="iam:CreateAccessKey"
            )
            events.append(event)

        if day_offset == 20:  # Later anomaly: bulk data access
            for i in range(500):
                anomaly_time = current_date.replace(hour=14, minute=30 + (i // 60), second=i % 60)
                event = generate_event(
                    "bob.jones", USERS["bob.jones"], anomaly_time,
                    is_anomaly=True, anomaly_api="s3:GetObject"
                )
                events.append(event)

        if day_offset == 25:  # Unknown external user
            for i in range(10):
                anomaly_time = current_date.replace(hour=2, minute=i, second=0)
                event = generate_event(
                    "unknown-user-ext",
                    {"role": "developer", "typical_hours": (0, 24), "api_frequency": "low"},
                    anomaly_time,
                    is_anomaly=True,
                    anomaly_api="sts:AssumeRole",
                    source_ip=random.choice(EXTERNAL_IPS)
                )
                events.append(event)

    print(f"Generated {len(events)} total events")
    return events

def save_to_duckdb(events: list, db_path: str = "cloudtrail_baseline.duckdb"):
    """Save events to DuckDB database."""

    # Remove existing file if present
    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()

    conn = duckdb.connect(db_path)

    # Create table with flattened structure for analysis
    conn.execute("""
        CREATE TABLE cloudtrail_events (
            event_id VARCHAR PRIMARY KEY,
            event_time VARCHAR,
            event_date DATE,
            event_hour INTEGER,
            event_source VARCHAR,
            event_name VARCHAR,
            aws_region VARCHAR,
            source_ip VARCHAR,
            user_agent VARCHAR,
            user_type VARCHAR,
            user_name VARCHAR,
            user_arn VARCHAR,
            account_id VARCHAR,
            request_id VARCHAR,
            error_code VARCHAR,
            error_message VARCHAR,
            read_only BOOLEAN,
            event_type VARCHAR,
            event_category VARCHAR
        )
    """)

    # Prepare data for bulk insert
    rows = []
    for event in events:
        event_time = event["eventTime"]
        event_date = event_time[:10]
        event_hour = int(event_time[11:13])

        rows.append((
            event["eventID"],
            event["eventTime"],
            event_date,
            event_hour,
            event["eventSource"],
            event["eventName"],
            event["awsRegion"],
            event["sourceIPAddress"],
            event["userAgent"],
            event["userIdentity"]["type"],
            event["userIdentity"]["userName"],
            event["userIdentity"]["arn"],
            event["userIdentity"]["accountId"],
            event["requestID"],
            event.get("errorCode"),
            event.get("errorMessage"),
            event["readOnly"],
            event["eventType"],
            event["eventCategory"]
        ))

    # Bulk insert using executemany
    conn.executemany("""
        INSERT INTO cloudtrail_events VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, rows)

    # Create useful indexes
    conn.execute("CREATE INDEX idx_user_name ON cloudtrail_events(user_name)")
    conn.execute("CREATE INDEX idx_event_name ON cloudtrail_events(event_name)")
    conn.execute("CREATE INDEX idx_event_date ON cloudtrail_events(event_date)")
    conn.execute("CREATE INDEX idx_source_ip ON cloudtrail_events(source_ip)")

    conn.close()

    print(f"Saved {len(events)} events to {db_path}")

def save_to_json(events: list, json_path: str = "cloudtrail_events.json"):
    """Save events to JSON (simulating S3/data lake pattern)."""
    with open(json_path, 'w') as f:
        json.dump(events, f, indent=2)
    print(f"Saved {len(events)} events to {json_path}")

if __name__ == "__main__":
    # Determine output directory (data/ relative to project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    # Generate data
    events = generate_dataset(days=30, events_per_day_base=1000)

    # Save to both formats in data/ directory
    save_to_duckdb(events, str(data_dir / "cloudtrail_baseline.duckdb"))
    save_to_json(events, str(data_dir / "cloudtrail_events.json"))

    print("\nDataset generation complete!")
    print(f"Output directory: {data_dir}")
    print("Injected anomalies:")
    print("  - Day 15: Off-hours credential creation (alice.smith, 3 AM)")
    print("  - Day 20: Bulk S3 data access (bob.jones, 500 events)")
    print("  - Day 25: Unknown external user AssumeRole attempts")
