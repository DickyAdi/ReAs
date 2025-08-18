from datetime import datetime

class ResetPasswordTemplate:
    """Generate email response for reset password request.

    Method:
        __call__
            Args:
                device (str) : Request user agent
                request_time (datetime) : Request time parsed
                reset_link (str) : reset link with token
            Return:
                str : Formatted HTML email response.
    """
    def __call__(self, device:str, request_time:datetime, reset_link:str):
        request_time = request_time.strftime("%H:%M; %d/%m/%Y")
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
    <h2>Password Reset Request</h2>
    <p>Hello,</p>
    <p>We received a request to reset your password from {device} at {request_time}. If you did not make this request, you can safely ignore this email.</p>
    <p>Click the button below to reset your password:</p>

    <p style="text-align: center; margin: 30px 0;">
      <a href="{reset_link}" class="button">Reset Password</a>
    </p>

    <div style="overflow-wrap: break-word;">
      <p>This link will expire in 15 minutes. If the button above doesn't work, copy and paste this URL into your browser:</p>
      <p><a href="{reset_link}">{reset_link}</a></p>
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
        return template.format(device=device, request_time=request_time, reset_link=reset_link)