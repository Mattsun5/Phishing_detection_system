import pandas as pd
import random

# Parameters
total_emails = 5000
phishing_emails = total_emails // 2
legit_emails = total_emails // 2

# Example templates for phishing and legitimate emails
phishing_templates = [
    "Your account has been suspended. Click here to verify: {}",
    "Urgent: Update your payment info at {}",
    "You've won a prize! Claim now: {}",
    "Security alert! Login here to secure your account: {}",
    "Action required: Confirm your credentials at {}"
]

legit_templates = [
    "Meeting scheduled for tomorrow at {}",
    "Here is the report you requested: {}",
    "Thank you for your purchase! Your order {} has shipped",
    "Reminder: Your appointment is on {}",
    "Monthly newsletter: {}"
]

urls = ["http://bit.ly/abc123", "https://secure-login.com", "http://phishingsite.net", "http://update-info.com", "https://example.com/info"]

# Generate emails
emails = []

for _ in range(phishing_emails):
    template = random.choice(phishing_templates)
    email_text = template.format(random.choice(urls))
    emails.append({"text": email_text, "label": "phishing"})

for _ in range(legit_emails):
    template = random.choice(legit_templates)
    email_text = template.format(random.choice(urls))
    emails.append({"text": email_text, "label": "legitimate"})

# Shuffle dataset
random.shuffle(emails)

# Create DataFrame
df = pd.DataFrame(emails)

# Save to CSV
df.to_csv("emails.csv", index=False)
print("emails.csv generated successfully with 5,000 emails!")
