# Graph Visualization Style Guide

## How to Change Graph Colors in Memgraph Lab

### Method 1: Using the Style Editor (Easiest)

1. **Open Memgraph Lab** in your browser:
   ```
   http://localhost:3000
   ```

2. **Click the Style button** (paint brush icon) in the top right corner

3. **Copy and paste** one of the style scripts from `graph_style.cypher`

4. **Click "Apply"** to see your changes

---

## Available Color Schemes

### 🔵 Blue (Professional)
- **Node Color:** `#3B82F6` (bright blue)
- **Border Color:** `#1E40AF` (dark blue)
- **Best for:** Professional presentations, business reports
- **File:** Use the default in `graph_style.cypher`

### 🟢 Green (Nature/Health)
- **Node Color:** `#10B981` (emerald green)
- **Border Color:** `#047857` (dark green)
- **Best for:** Healthcare, biology, environmental themes
- **File:** Uncomment Option 2 in `graph_style.cypher`

### 🟣 Purple (Creative)
- **Node Color:** `#8B5CF6` (vibrant purple)
- **Border Color:** `#5B21B6` (deep purple)
- **Best for:** Creative projects, research presentations
- **File:** Uncomment Option 3 in `graph_style.cypher`

### 🔷 Teal (Modern)
- **Node Color:** `#14B8A6` (bright teal)
- **Border Color:** `#0D9488` (dark teal)
- **Best for:** Modern UI, tech presentations
- **File:** Uncomment Option 4 in `graph_style.cypher`

### 🟠 Orange (Warm)
- **Node Color:** `#F97316` (bright orange)
- **Border Color:** `#C2410C` (dark orange)
- **Best for:** Engaging presentations, highlighting insights
- **File:** Uncomment Option 5 in `graph_style.cypher`

---

## Advanced: Color by Node Type

Want to color-code different entity types? Use the **Custom** option at the bottom of `graph_style.cypher`:

- **Drugs:** Blue
- **Diseases:** Red
- **Proteins:** Green
- **Pathways:** Purple

This makes it easy to identify different types of nodes at a glance!

---

## Style Properties Explained

```cypher
@NodeStyle {
  size: 40;              // Node diameter in pixels
  border-width: 2px;     // Border thickness
  border-color: #1E40AF; // Darker shade for borders
  color: #3B82F6;        // Main node color
  label: {
    font-size: 12px;     // Label text size
    color: #1F2937;      // Label text color
  }
}

@EdgeStyle {
  width: 2px;            // Edge line thickness
  color: #6B7280;        // Gray for edges (consistent)
  label: {
    font-size: 10px;     // Relationship label size
    color: #4B5563;      // Label text color
  }
}
```

---

## Quick Color Reference

| Color Name | Node Color | Border Color | Hex Code |
|------------|-----------|--------------|----------|
| Red (Original) | `#DC2626` | `#991B1B` | Crimson |
| Blue | `#3B82F6` | `#1E40AF` | Sky blue |
| Green | `#10B981` | `#047857` | Emerald |
| Purple | `#8B5CF6` | `#5B21B6` | Violet |
| Teal | `#14B8A6` | `#0D9488` | Cyan-green |
| Orange | `#F97316` | `#C2410C` | Bright orange |

---

## Custom Colors

Want a specific color? Modify the hex codes:

```cypher
@NodeStyle {
  color: #YOUR_COLOR_HERE;        // Main node color
  border-color: #DARKER_SHADE;    // Border (usually darker)
}
```

**Pro tip:** Use a color picker tool to find hex codes:
- [Coolors.co](https://coolors.co)
- [HTML Color Codes](https://htmlcolorcodes.com)

---

## Keeping the Same Style

All color schemes maintain the original design:
- ✅ Same node size (40px)
- ✅ Same border width (2px)
- ✅ Same edge color (gray)
- ✅ Same label font size
- ✅ Same layout algorithm

**Only the node colors change!**

---

## Troubleshooting

### Style not applying?
1. Make sure Memgraph Lab is running
2. Check for syntax errors in the style script
3. Try refreshing the page

### Colors look wrong?
1. Verify you're using the correct hex codes
2. Check that `#` is included before color codes
3. Ensure semicolons (`;`) are at the end of each property

### Want to reset?
Click the "Reset to default" button in the Style editor

---

## Export Your Visualization

Once you have the perfect colors:

1. **Screenshot:** Use your browser's screenshot tool
2. **Export:** Click "Export" in Memgraph Lab
3. **Save style:** Copy the style script to save for later

---

## Files in This Project

- `graph_style.cypher` - Main style script with all options
- `graph_style_blue.json` - Blue color scheme (JSON format)
- `graph_style_green.json` - Green color scheme (JSON format)
- `graph_style_purple.json` - Purple color scheme (JSON format)
- `graph_style_teal.json` - Teal color scheme (JSON format)
- `GRAPH_STYLE_GUIDE.md` - This guide!

---

## Quick Start

```bash
# 1. Open Memgraph Lab
open http://localhost:3000

# 2. Load your data (if not already loaded)
# Run your Cypher queries to create the graph

# 3. Apply style
# Click Style button → Paste from graph_style.cypher → Apply

# 4. Enjoy your new colors! 🎨
```
