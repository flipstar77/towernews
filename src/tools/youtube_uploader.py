"""YouTube Uploader - Uploads videos directly to YouTube via API."""

import os
import pickle
from pathlib import Path
from datetime import datetime

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# OAuth scopes for YouTube upload
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class YouTubeUploader:
    """Uploads videos to YouTube via the Data API v3."""

    def __init__(self, config: dict):
        self.config = config
        self.channel_name = config.get("channel", {}).get("name", "Tower News")
        self.credentials_path = Path("config/youtube_credentials.json")
        self.token_path = Path("config/youtube_token.pickle")
        self._youtube = None

    def _get_authenticated_service(self):
        """Get authenticated YouTube service."""
        if self._youtube:
            return self._youtube

        creds = None

        # Load existing token
        if self.token_path.exists():
            with open(self.token_path, "rb") as token:
                creds = pickle.load(token)

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"YouTube credentials not found at {self.credentials_path}\n"
                        "Please download OAuth credentials from Google Cloud Console:\n"
                        "1. Go to https://console.cloud.google.com/\n"
                        "2. Create/select project, enable YouTube Data API v3\n"
                        "3. Create OAuth 2.0 credentials (Desktop app)\n"
                        "4. Download and save as config/youtube_credentials.json"
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path),
                    SCOPES
                )
                creds = flow.run_local_server(port=8080)

            # Save credentials for next run
            with open(self.token_path, "wb") as token:
                pickle.dump(creds, token)
            print("[YouTubeUploader] Credentials saved for future use")

        self._youtube = build("youtube", "v3", credentials=creds)
        return self._youtube

    def upload(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: list[str] = None,
        category_id: str = "20",  # Gaming category
        privacy_status: str = "private",  # Start as private for safety
        publish_at: str = None  # ISO 8601 datetime for scheduled publishing
    ) -> dict:
        """
        Upload a video to YouTube.

        Args:
            video_path: Path to the video file
            title: Video title (max 100 chars)
            description: Video description (max 5000 chars)
            tags: List of tags
            category_id: YouTube category (20 = Gaming)
            privacy_status: "private", "public", or "unlisted"
            publish_at: Schedule publish time (ISO 8601, only for private videos)

        Returns:
            Dictionary with video_id, url, and success status
        """
        if not Path(video_path).exists():
            return {"success": False, "error": f"Video file not found: {video_path}"}

        try:
            youtube = self._get_authenticated_service()

            # Prepare video metadata
            body = {
                "snippet": {
                    "title": title[:100],  # Max 100 chars
                    "description": description[:5000],  # Max 5000 chars
                    "tags": tags or [],
                    "categoryId": category_id,
                    "defaultLanguage": "en",
                    "defaultAudioLanguage": "en"
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False
                }
            }

            # Add scheduled publish time if provided
            if publish_at and privacy_status == "private":
                body["status"]["publishAt"] = publish_at

            # Create media upload
            media = MediaFileUpload(
                video_path,
                mimetype="video/mp4",
                resumable=True,
                chunksize=1024 * 1024 * 10  # 10MB chunks
            )

            # Execute upload
            print(f"[YouTubeUploader] Starting upload: {title[:50]}...")
            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    print(f"[YouTubeUploader] Upload progress: {progress}%")

            video_id = response["id"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            print(f"[YouTubeUploader] Upload complete!")
            print(f"[YouTubeUploader] Video URL: {video_url}")

            return {
                "success": True,
                "video_id": video_id,
                "url": video_url,
                "privacy_status": privacy_status,
                "title": title
            }

        except Exception as e:
            print(f"[YouTubeUploader] Upload failed: {e}")
            return {"success": False, "error": str(e)}

    def upload_from_metadata(
        self,
        video_path: str,
        metadata_path: str,
        privacy_status: str = "private"
    ) -> dict:
        """
        Upload video using metadata from generated metadata file.

        Args:
            video_path: Path to video file
            metadata_path: Path to metadata.txt file
            privacy_status: "private", "public", or "unlisted"

        Returns:
            Upload result dictionary
        """
        # Parse metadata file
        title = ""
        description = ""
        tags = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract sections
        if "=== TITLE ===" in content:
            title_section = content.split("=== TITLE ===")[1].split("===")[0]
            title = title_section.strip()

        if "=== DESCRIPTION ===" in content:
            desc_section = content.split("=== DESCRIPTION ===")[1].split("=== TAGS ===")[0]
            description = desc_section.strip()

        if "=== TAGS ===" in content:
            tags_section = content.split("=== TAGS ===")[1].split("===")[0]
            tags = [t.strip() for t in tags_section.strip().split(",")]

        return self.upload(
            video_path=video_path,
            title=title,
            description=description,
            tags=tags,
            privacy_status=privacy_status
        )

    def run(self, **kwargs) -> dict:
        """Tool interface for pipeline."""
        video_path = kwargs.get("video_path")
        metadata_path = kwargs.get("metadata_path")
        privacy_status = kwargs.get("privacy_status", "private")

        if metadata_path:
            return self.upload_from_metadata(video_path, metadata_path, privacy_status)
        else:
            return self.upload(
                video_path=video_path,
                title=kwargs.get("title", f"{self.channel_name} - {datetime.now().strftime('%Y-%m-%d')}"),
                description=kwargs.get("description", ""),
                tags=kwargs.get("tags", []),
                privacy_status=privacy_status
            )
