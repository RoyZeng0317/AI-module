import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing() as d:
    d += elm.SourceV().up().label('5V (Arduino Pin)')
    d += elm.Resistor().right().label('10kΩ')
    d += elm.LED().down().label('Red LED')
    d += elm.Line().left()