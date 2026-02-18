# Outfit Finder Extension (Starter)

This extension lets you drag-select an outfit region directly on the page, then sends that crop to your backend `/breakdown` endpoint.

## Load in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select this folder: `extension/`

## Configure backend URL

1. Open extension popup
2. Click **Settings**
3. Set API base URL (local example: `http://127.0.0.1:8000`)

## Backend requirements

Your backend must expose:

- `POST /breakdown?max_items=3&k_per_item=8`
- accept multipart file field named `file`
- return JSON with `groups` where each group has `category`, `confidence`, and `results`

## Local test flow

1. Start backend:
   - `uvicorn api:app --reload`
2. Open any page/video frame in Chrome
3. Open extension popup -> **Select Region On Page**
4. Drag on the page to select the outfit piece
5. Popup may close automatically (Chrome behavior), reopen it
6. Click **Search Selected Area**
