
class VerifyEmailTemplate:
    def __call__(self, verify_link:str):
        template = """
            <!DOCTYPE html>
<html lang="en" style="margin: 0; padding: 0;">
<head>
  <meta charset="UTF-8">
  <title>Password Reset</title>
  <style>
    a.button {{
      background-color: #4CAF50;
      color: white !important;
      padding: 12px 24px;
      text-decoration: none;
      border-radius: 4px;
      display: inline-block;
    }}
    .container {{
      width: 100%;
      max-width: 600px;
      margin: auto;
      padding: 20px;
      font-family: Arial, sans-serif;
      color: #333;
    }}
    .footer {{
      font-size: 12px;
      color: #888;
      text-align: center;
      margin-top: 30px;
    }}
  </style>
</head>
<body style="background-color: #f9f9f9; padding: 20px;">

  <div class="container" style="background-color: #ffffff; border-radius: 6px; box-shadow: 0 0 10px rgba(0,0,0,0.05);">
    <h2>Welcome to Review Assistant (ReAs)</h2>
    <p>Hi there,</p>
    <p>Thank you for signing up for ReAs, We're excited to have you onboard!</p>
    <p>To get started and ensure the security of your account, please verify your email address by clicking the button below;</p>

    <p style="text-align: center; margin: 30px 0;">
      <a href="{verify_link}" class="button">Verify</a>
    </p>
    <div style="overflow-wrap: break-word;">
      <p>This link will expire in 15 minutes. If the button above doesn't work, copy and paste this URL into your browser:</p>
      <p><a href="{verify_link}">{verify_link}</a></p>
    </div>

    <p>Thanks,<br>The ReAs Team</p>

    <div class="footer">
      If you have any questions, contact us at support@reas.example.com<br>
      &copy; 2025 ReAs, All rights reserved.
    </div>
  </div>

</body>
</html>
        """
        return template.format(verify_link=verify_link)