// =====================================================================
// MEMGRAPH LAB GRAPH STYLE SCRIPT
// =====================================================================
// Copy this style script into Memgraph Lab to change visualization colors
//
// To apply:
// 1. Open Memgraph Lab (http://localhost:3000)
// 2. Click the "Style" button (paint brush icon) in the top right
// 3. Paste the style script below
// 4. Click "Apply"
//
// Choose one of the color schemes below:
// =====================================================================


// =====================================================================
// OPTION 1: BLUE (Professional)
// =====================================================================
@NodeStyle {
  size: 40;
  border-width: 2px;
  border-color: #1E40AF;
  color: #3B82F6;
  label: {
    font-size: 12px;
    color: #1F2937;
  }
}

@EdgeStyle {
  width: 2px;
  color: #6B7280;
  label: {
    font-size: 10px;
    color: #4B5563;
  }
}


// =====================================================================
// OPTION 2: GREEN (Nature/Health)
// Uncomment to use (remove // before each line)
// =====================================================================
/*
@NodeStyle {
  size: 40;
  border-width: 2px;
  border-color: #047857;
  color: #10B981;
  label: {
    font-size: 12px;
    color: #1F2937;
  }
}

@EdgeStyle {
  width: 2px;
  color: #6B7280;
  label: {
    font-size: 10px;
    color: #4B5563;
  }
}
*/


// =====================================================================
// OPTION 3: PURPLE (Creative)
// Uncomment to use
// =====================================================================
/*
@NodeStyle {
  size: 40;
  border-width: 2px;
  border-color: #5B21B6;
  color: #8B5CF6;
  label: {
    font-size: 12px;
    color: #1F2937;
  }
}

@EdgeStyle {
  width: 2px;
  color: #6B7280;
  label: {
    font-size: 10px;
    color: #4B5563;
  }
}
*/


// =====================================================================
// OPTION 4: TEAL (Modern)
// Uncomment to use
// =====================================================================
/*
@NodeStyle {
  size: 40;
  border-width: 2px;
  border-color: #0D9488;
  color: #14B8A6;
  label: {
    font-size: 12px;
    color: #1F2937;
  }
}

@EdgeStyle {
  width: 2px;
  color: #6B7280;
  label: {
    font-size: 10px;
    color: #4B5563;
  }
}
*/


// =====================================================================
// OPTION 5: ORANGE (Warm)
// Uncomment to use
// =====================================================================
/*
@NodeStyle {
  size: 40;
  border-width: 2px;
  border-color: #C2410C;
  color: #F97316;
  label: {
    font-size: 12px;
    color: #1F2937;
  }
}

@EdgeStyle {
  width: 2px;
  color: #6B7280;
  label: {
    font-size: 10px;
    color: #4B5563;
  }
}
*/


// =====================================================================
// CUSTOM: Different Colors by Node Type
// Uncomment to color-code by entity type
// =====================================================================
/*
@NodeStyle {
  size: 40;
  border-width: 2px;
  label: {
    font-size: 12px;
    color: #1F2937;
  }
}

@NodeStyle HasLabel(node, "drug") {
  color: #3B82F6;
  border-color: #1E40AF;
}

@NodeStyle HasLabel(node, "disease") {
  color: #EF4444;
  border-color: #B91C1C;
}

@NodeStyle HasLabel(node, "gene/protein") {
  color: #10B981;
  border-color: #047857;
}

@NodeStyle HasLabel(node, "pathway") {
  color: #8B5CF6;
  border-color: #5B21B6;
}

@NodeStyle Property(node, "node_type") == "drug" {
  color: #3B82F6;
  border-color: #1E40AF;
}

@NodeStyle Property(node, "node_type") == "disease" {
  color: #EF4444;
  border-color: #B91C1C;
}

@NodeStyle Property(node, "node_type") == "gene/protein" {
  color: #10B981;
  border-color: #047857;
}

@NodeStyle Property(node, "node_type") == "pathway" {
  color: #8B5CF6;
  border-color: #5B21B6;
}

@EdgeStyle {
  width: 2px;
  color: #6B7280;
  label: {
    font-size: 10px;
    color: #4B5563;
  }
}
*/
