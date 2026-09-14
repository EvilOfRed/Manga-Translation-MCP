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
        ],
        "layout":"horizontal"
      },
      {
        "original": "...",
        "translation": "...",
        "boxes": [
          { "coords": [.., .., .., ..], "label": "text_bubble" }
        ],
        "layout":"horizontal"
      },
      {
        "original": "...",
        "translation": "...",
        "boxes": [],
        "layout":"horizontal"
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
        ],
        "layout":"horizontal"
      },
      {
        "original": "...",
        "translation": "...",
        "boxes": [],
        "layout":"horizontal"
      },
      {
        "original": "...",
        "translation": "...",
        "boxes": [
          { "coords": [.., .., .., ..], "label": "text_bubble" }
        ],
        "layout":"horizontal"
      }
    ]
  }
]