# JSON 数据示例

```json
[
  {
    "page": "1",
    "context": [
      {
        "original": "...",
        "translation": "...",
        "boxes": [
          { "coords": [.., .., .., ..], "label": "text_bubble" },
          { "coords": [.., .., .., ..], "label": "text_free" }
        ]
      },
      {
        "original": "...",
        "translation": "...",
        "boxes": [
          { "coords": [.., .., .., ..], "label": "text_bubble" }
        ]
      },
      {
        "original": "...",
        "translation": "...",
        "boxes": []
      }
    ]
  },
  {
    "page": "2",
    "context": [
      {
        "original": "...",
        "translation": "...",
        "boxes": [
          { "coords": [.., .., .., ..], "label": "text_bubble" },
          { "coords": [.., .., .., ..], "label": "text_free" }
        ]
      },
      {
        "original": "...",
        "translation": "...",
        "boxes": []
      },
      {
        "original": "...",
        "translation": "...",
        "boxes": [
          { "coords": [.., .., .., ..], "label": "text_bubble" }
        ]
      }
    ]
  }
]

```