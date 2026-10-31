; ============================================================
; Python-used SKILL template for octagonal transformer
;
; This file is read by engine.py:
;   transformer.template_file: ".../templates/carrier_transformer/transformer_layout.il"
;
; Python replaces:
;   {{LIB_NAME}}, {{CELL_NAME}},
;   {{PRI_LAYER}}, {{SEC_LAYER}}, {{GND_LAYER}},
;   {{PRI_INNER_R}}, {{PRI_WIDTH}},
;   {{SEC_INNER_R}}, {{SEC_WIDTH}}
;
; Other dimensions are derived proportionally inside SKILL.
; ============================================================

procedure( build_octagon_transformer()
let( (cv obj
      libName cellName viewName
      priLayerName secLayerName gndLayerName
      priLayer secLayer gndLayer
      pinLayerPri pinLayerSec pinLayerGnd
      cx cy pi angles a x y
      priInnerR secInnerR
      priWidth secWidth
      priOuterR secOuterR
      priCenterR secCenterR
      ptsOuter ptsInner pts
      pinSize pinBox
      termBendPct
      priTermBendLen secTermBendLen
      pinYOffsetPct pinYOffset
      priUpperEndY priLowerEndY
      secUpperEndY secLowerEndY
      priRightOuterX priRightInnerX
      secLeftOuterX secLeftInnerX
      fig
      gndTraceWidth
      gndRefR gndOuterHalf gndInnerHalf gndMargin
      gndNet gndFig
      netP1 netP2 netP3 netP4
      x1 x2 y1 y2
      p1x p1y p2x p2y p3x p3y p4x p4y
      centerTapWidth
      tapX1 tapX2 tapY1 tapY2
      tapCapX1 tapCapX2 tapCapY1 tapCapY2
      ctPinX ctPinY1 ctPinY2
      maxWidth minWidth)

  ; ==========================================
  ; USER CONFIGURATION
  ; Only change these 4 geometry parameters
  ; ==========================================
  libName  = "{{LIB_NAME}}"
  cellName = "{{CELL_NAME}}"
  viewName = "layout"

  gndLayerName = "{{GND_LAYER}}"
  priLayerName = "{{PRI_LAYER}}"
  secLayerName = "{{SEC_LAYER}}"

  cx = 0.0
  cy = 0.0

  ; -------- ONLY CHANGE THESE FOUR --------
  priInnerR = {{PRI_INNER_R}}
  priWidth  = {{PRI_WIDTH}}

  secInnerR = {{SEC_INNER_R}}
  secWidth  = {{SEC_WIDTH}}
  ; ----------------------------------------

  ; ==========================================
  ; DERIVED PROPORTIONAL PARAMETERS
  ; Do not need to manually tune these
  ; ==========================================
  maxWidth = max(priWidth secWidth)
  minWidth = min(priWidth secWidth)

  ; GND loop width follows primary coil width
  gndTraceWidth = priWidth

  ; Pin size scales with metal width
  pinSize = 0.30 * minWidth

  ; Keep minimum pin size reasonable
  when(pinSize < 1.0
    pinSize = 1.0
  )

  ; Terminal bend and y-offset scale with geometry
  termBendPct = 0.8
  pinYOffsetPct = 0.35
  pinYOffset = maxWidth * pinYOffsetPct

  ; Center tap metal is entirely on the primary layer
  centerTapWidth = priWidth

  ; ==========================================
  ; OPEN OR CREATE LAYOUT
  ; ==========================================
  cv = dbOpenCellViewByType(
         libName
         cellName
         viewName
         "maskLayout"
         "a"
       )

  unless(cv
    error("Cannot open/create layout cell %s/%s/%s\n"
          libName cellName viewName)
  )

  ; ==========================================
  ; CLEAR OLD CONTENT
  ; ==========================================
  foreach(obj cv~>shapes
    dbDeleteObject(obj)
  )

  foreach(obj cv~>instances
    dbDeleteObject(obj)
  )

  foreach(obj cv~>labels
    dbDeleteObject(obj)
  )

  foreach(obj cv~>vias
    dbDeleteObject(obj)
  )

  foreach(obj cv~>terminals
    dbDeleteObject(obj)
  )

  foreach(obj cv~>nets
    dbDeleteObject(obj)
  )

  ; ==========================================
  ; LAYERS
  ; ==========================================
  priLayer = list(priLayerName "drawing")
  secLayer = list(secLayerName "drawing")
  gndLayer = list(gndLayerName "drawing")

  pinLayerPri = list(priLayerName "pin")
  pinLayerSec = list(secLayerName "pin")
  pinLayerGnd = list(gndLayerName "pin")

  pi = 3.141592653589793

  angles = list(
      22.5 67.5 112.5 157.5
      202.5 247.5 292.5 337.5
  )

  ; ==========================================
  ; CREATE NETS
  ; ==========================================
  gndNet = dbCreateNet(cv "GND")
  dbCreateTerm(gndNet "GND" "inputOutput")

  netP1 = dbCreateNet(cv "P1")
  dbCreateTerm(netP1 "P1" "inputOutput")

  netP2 = dbCreateNet(cv "P2")
  dbCreateTerm(netP2 "P2" "inputOutput")

  netP3 = dbCreateNet(cv "P3")
  dbCreateTerm(netP3 "P3" "inputOutput")

  netP4 = dbCreateNet(cv "P4")
  dbCreateTerm(netP4 "P4" "inputOutput")

  ; ==========================================
  ; PRIMARY COIL ON metal2
  ; ==========================================
  priOuterR  = priInnerR + priWidth / cos(22.5*pi/180.0)
  priCenterR = (priInnerR + priOuterR) / 2.0

  priTermBendLen = priWidth * termBendPct

  priUpperEndY = cy + priOuterR*sin(22.5*pi/180.0) - priTermBendLen + pinYOffset
  priLowerEndY = cy + priOuterR*sin(337.5*pi/180.0) + priTermBendLen - pinYOffset

  ptsOuter = nil

  x = cx + priOuterR*cos(22.5*pi/180.0)
  y = priUpperEndY
  ptsOuter = append(ptsOuter list(list(x y)))

  foreach(a angles
    x = cx + priOuterR*cos(a*pi/180.0)
    y = cy + priOuterR*sin(a*pi/180.0)
    ptsOuter = append(ptsOuter list(list(x y)))
  )

  x = cx + priOuterR*cos(337.5*pi/180.0)
  y = priLowerEndY
  ptsOuter = append(ptsOuter list(list(x y)))

  ptsInner = nil

  x = cx + priInnerR*cos(337.5*pi/180.0)
  y = priLowerEndY
  ptsInner = append(ptsInner list(list(x y)))

  foreach(a reverse(angles)
    x = cx + priInnerR*cos(a*pi/180.0)
    y = cy + priInnerR*sin(a*pi/180.0)
    ptsInner = append(ptsInner list(list(x y)))
  )

  x = cx + priInnerR*cos(22.5*pi/180.0)
  y = priUpperEndY
  ptsInner = append(ptsInner list(list(x y)))

  pts = append(ptsOuter ptsInner)
  dbCreatePolygon(cv priLayer pts)

  ; ==========================================
  ; SECONDARY COIL ON metal3
  ; ==========================================
  secOuterR  = secInnerR + secWidth / cos(22.5*pi/180.0)
  secCenterR = (secInnerR + secOuterR) / 2.0

  secTermBendLen = secWidth * termBendPct

  secUpperEndY = cy + secOuterR*sin(157.5*pi/180.0) - secTermBendLen + pinYOffset
  secLowerEndY = cy + secOuterR*sin(202.5*pi/180.0) + secTermBendLen - pinYOffset

  ptsOuter = nil

  x = cx + secOuterR*cos(202.5*pi/180.0)
  y = secLowerEndY
  ptsOuter = append(ptsOuter list(list(x y)))

  foreach(a angles
    x = cx + secOuterR*cos((a+180.0)*pi/180.0)
    y = cy + secOuterR*sin((a+180.0)*pi/180.0)
    ptsOuter = append(ptsOuter list(list(x y)))
  )

  x = cx + secOuterR*cos(157.5*pi/180.0)
  y = secUpperEndY
  ptsOuter = append(ptsOuter list(list(x y)))

  ptsInner = nil

  x = cx + secInnerR*cos(157.5*pi/180.0)
  y = secUpperEndY
  ptsInner = append(ptsInner list(list(x y)))

  foreach(a reverse(angles)
    x = cx + secInnerR*cos((a+180.0)*pi/180.0)
    y = cy + secInnerR*sin((a+180.0)*pi/180.0)
    ptsInner = append(ptsInner list(list(x y)))
  )

  x = cx + secInnerR*cos(202.5*pi/180.0)
  y = secLowerEndY
  ptsInner = append(ptsInner list(list(x y)))

  pts = append(ptsOuter ptsInner)
  dbCreatePolygon(cv secLayer pts)

  ; ==========================================
  ; PROPORTIONAL RECTANGULAR GND RING ON metal1
  ; ==========================================
  gndRefR = max(priOuterR secOuterR)

  ; Margin scales with transformer size and metal width
  gndMargin = max(2.0*maxWidth 0.20*gndRefR)

  gndOuterHalf = gndRefR + gndMargin
  gndInnerHalf = gndOuterHalf - gndTraceWidth

  ; left GND metal
  dbCreateRect(
    cv
    gndLayer
    list(
      list(cx-gndOuterHalf cy-gndOuterHalf)
      list(cx-gndInnerHalf cy+gndOuterHalf)
    )
  )

  ; right GND metal
  dbCreateRect(
    cv
    gndLayer
    list(
      list(cx+gndInnerHalf cy-gndOuterHalf)
      list(cx+gndOuterHalf cy+gndOuterHalf)
    )
  )

  ; top GND metal
  dbCreateRect(
    cv
    gndLayer
    list(
      list(cx-gndOuterHalf cy+gndInnerHalf)
      list(cx+gndOuterHalf cy+gndOuterHalf)
    )
  )

  ; bottom GND metal
  dbCreateRect(
    cv
    gndLayer
    list(
      list(cx-gndOuterHalf cy-gndOuterHalf)
      list(cx+gndOuterHalf cy-gndInnerHalf)
    )
  )

  ; ==========================================
  ; LEFT-SIDE T-SHAPED PRIMARY CENTER TAP
  ; Entire center tap geometry is on the primary layer.
  ; ==========================================

  ; horizontal part from the left GND side to the primary coil
  tapX1 = cx - gndOuterHalf
  tapX2 = cx - priOuterR*cos(22.5*pi/180.0) + 0.20*priWidth

  tapY1 = cy - centerTapWidth/2.0
  tapY2 = cy + centerTapWidth/2.0

  dbCreateRect(
    cv
    priLayer
    list(
      list(tapX1 tapY1)
      list(tapX2 tapY2)
    )
  )

  ; vertical part of T on primary layer
  ; same length as full square GND loop side
  tapCapX1 = cx - gndOuterHalf
  tapCapX2 = cx - gndInnerHalf

  tapCapY1 = cy - gndOuterHalf
  tapCapY2 = cy + gndOuterHalf

  dbCreateRect(
    cv
    priLayer
    list(
      list(tapCapX1 tapCapY1)
      list(tapCapX2 tapCapY2)
    )
  )

  ; ==========================================
  ; CENTER-TAP SHORT PINS AT TWO ENDS
  ; 2 pins on primary layer
  ; 2 overlapping pins on GND layer metal1
  ; ==========================================

  ctPinX  = (tapCapX1 + tapCapX2) / 2.0

  ; pins at top and bottom ends of center tap
  ctPinY1 = tapCapY2 - pinSize
  ctPinY2 = tapCapY1 + pinSize

  ; upper-end center-tap primary-layer pin
  pinBox = list(
    list(ctPinX-pinSize ctPinY1-pinSize)
    list(ctPinX+pinSize ctPinY1+pinSize)
  )

  fig = dbCreateRect(cv pinLayerPri pinBox)
  dbCreatePin(gndNet fig)

  dbCreateLabel(
    cv
    pinLayerPri
    list(ctPinX ctPinY1)
    "PRI_CT_GND"
    "centerCenter"
    "R0"
    "stick"
    2.0
  )

  ; upper-end overlapping GND-layer pin
  gndFig = dbCreateRect(cv pinLayerGnd pinBox)
  dbCreatePin(gndNet gndFig)

  dbCreateLabel(
    cv
    pinLayerGnd
    list(ctPinX ctPinY1)
    "GND"
    "centerCenter"
    "R0"
    "stick"
    2.0
  )

  ; lower-end center-tap primary-layer pin
  pinBox = list(
    list(ctPinX-pinSize ctPinY2-pinSize)
    list(ctPinX+pinSize ctPinY2+pinSize)
  )

  fig = dbCreateRect(cv pinLayerPri pinBox)
  dbCreatePin(gndNet fig)

  dbCreateLabel(
    cv
    pinLayerPri
    list(ctPinX ctPinY2)
    "PRI_CT_GND"
    "centerCenter"
    "R0"
    "stick"
    2.0
  )

  ; lower-end overlapping GND-layer pin
  gndFig = dbCreateRect(cv pinLayerGnd pinBox)
  dbCreatePin(gndNet gndFig)

  dbCreateLabel(
    cv
    pinLayerGnd
    list(ctPinX ctPinY2)
    "GND"
    "centerCenter"
    "R0"
    "stick"
    2.0
  )

  ; ==========================================
  ; EXTEND metal2/metal3 TERMINALS INTO GND-PIN AREA
  ; Extension width equals the coil width it extends.
  ; ==========================================

  priRightOuterX = cx + priOuterR*cos(22.5*pi/180.0)
  priRightInnerX = cx + priInnerR*cos(22.5*pi/180.0)

  secLeftOuterX = cx + secOuterR*cos(157.5*pi/180.0)
  secLeftInnerX = cx + secInnerR*cos(157.5*pi/180.0)

  ; Port centers overlap with GND pin centers
  p1x = cx + gndInnerHalf + gndTraceWidth/2.0
  p1y = priUpperEndY

  p2x = cx + gndInnerHalf + gndTraceWidth/2.0
  p2y = priLowerEndY

  p3x = cx - gndInnerHalf - gndTraceWidth/2.0
  p3y = secUpperEndY

  p4x = cx - gndInnerHalf - gndTraceWidth/2.0
  p4y = secLowerEndY

  ; P1 upper-right metal2 extension
  ; width = priWidth
  x1 = priRightInnerX
  x2 = p1x + pinSize
  y1 = priUpperEndY - priWidth/2.0
  y2 = priUpperEndY + priWidth/2.0

  dbCreateRect(
    cv
    priLayer
    list(
      list(x1 y1)
      list(x2 y2)
    )
  )

  ; P2 lower-right metal2 extension
  ; width = priWidth
  x1 = priRightInnerX
  x2 = p2x + pinSize
  y1 = priLowerEndY - priWidth/2.0
  y2 = priLowerEndY + priWidth/2.0

  dbCreateRect(
    cv
    priLayer
    list(
      list(x1 y1)
      list(x2 y2)
    )
  )

  ; P3 upper-left metal3 extension
  ; width = secWidth
  x1 = p3x - pinSize
  x2 = secLeftInnerX
  y1 = secUpperEndY - secWidth/2.0
  y2 = secUpperEndY + secWidth/2.0

  dbCreateRect(
    cv
    secLayer
    list(
      list(x1 y1)
      list(x2 y2)
    )
  )

  ; P4 lower-left metal3 extension
  ; width = secWidth
  x1 = p4x - pinSize
  x2 = secLeftInnerX
  y1 = secLowerEndY - secWidth/2.0
  y2 = secLowerEndY + secWidth/2.0

  dbCreateRect(
    cv
    secLayer
    list(
      list(x1 y1)
      list(x2 y2)
    )
  )

  ; ==========================================
  ; SIGNAL PINS AND LOCAL OVERLAPPING GND PINS
  ; 4 signal pins + 4 GND pins
  ; ==========================================

  ; P1 and overlapping GND
  pinBox = list(
    list(p1x-pinSize p1y-pinSize)
    list(p1x+pinSize p1y+pinSize)
  )

  fig = dbCreateRect(cv pinLayerPri pinBox)
  dbCreatePin(netP1 fig)
  dbCreateLabel(cv pinLayerPri list(p1x p1y) "P1" "centerCenter" "R0" "stick" 2.0)

  gndFig = dbCreateRect(cv pinLayerGnd pinBox)
  dbCreatePin(gndNet gndFig)
  dbCreateLabel(cv pinLayerGnd list(p1x p1y) "GND" "centerCenter" "R0" "stick" 2.0)

  ; P2 and overlapping GND
  pinBox = list(
    list(p2x-pinSize p2y-pinSize)
    list(p2x+pinSize p2y+pinSize)
  )

  fig = dbCreateRect(cv pinLayerPri pinBox)
  dbCreatePin(netP2 fig)
  dbCreateLabel(cv pinLayerPri list(p2x p2y) "P2" "centerCenter" "R0" "stick" 2.0)

  gndFig = dbCreateRect(cv pinLayerGnd pinBox)
  dbCreatePin(gndNet gndFig)
  dbCreateLabel(cv pinLayerGnd list(p2x p2y) "GND" "centerCenter" "R0" "stick" 2.0)

  ; P3 and overlapping GND
  pinBox = list(
    list(p3x-pinSize p3y-pinSize)
    list(p3x+pinSize p3y+pinSize)
  )

  fig = dbCreateRect(cv pinLayerSec pinBox)
  dbCreatePin(netP3 fig)
  dbCreateLabel(cv pinLayerSec list(p3x p3y) "P3" "centerCenter" "R0" "stick" 2.0)

  gndFig = dbCreateRect(cv pinLayerGnd pinBox)
  dbCreatePin(gndNet gndFig)
  dbCreateLabel(cv pinLayerGnd list(p3x p3y) "GND" "centerCenter" "R0" "stick" 2.0)

  ; P4 and overlapping GND
  pinBox = list(
    list(p4x-pinSize p4y-pinSize)
    list(p4x+pinSize p4y+pinSize)
  )

  fig = dbCreateRect(cv pinLayerSec pinBox)
  dbCreatePin(netP4 fig)
  dbCreateLabel(cv pinLayerSec list(p4x p4y) "P4" "centerCenter" "R0" "stick" 2.0)

  gndFig = dbCreateRect(cv pinLayerGnd pinBox)
  dbCreatePin(gndNet gndFig)
  dbCreateLabel(cv pinLayerGnd list(p4x p4y) "GND" "centerCenter" "R0" "stick" 2.0)

  ; ==========================================
  ; SAVE AND CLOSE
  ; ==========================================
  dbSave(cv)
  dbClose(cv)

  printf("Created proportional transformer layout\n")
  printf("Only 4 geometry parameters need editing: priInnerR, priWidth, secInnerR, secWidth\n")
  printf("GND loop width = primary width = %.3f um\n" gndTraceWidth)
  printf("Primary extension width = %.3f um\n" priWidth)
  printf("Secondary extension width = %.3f um\n" secWidth)
  printf("Center tap: primary left side on %s\n" priLayerName)
  printf("Pin shapes: 4 signal + 4 local GND + 2 CT primary + 2 CT GND = 12\n")
  printf("Cell: %s/%s/%s\n" libName cellName viewName)
)
)

build_octagon_transformer()
