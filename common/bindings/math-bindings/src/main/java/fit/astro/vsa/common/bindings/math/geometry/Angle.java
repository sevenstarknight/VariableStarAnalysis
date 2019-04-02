/*
 * Copyright (C) 2012 United States Government as represented by the Administrator of the
 * National Aeronautics and Space Administration.
 * All Rights Reserved.
 */
package fit.astro.vsa.common.bindings.math.geometry;

import java.io.Serializable;

/**
 * Represents a geometric angle. Instances of <code>Angle</code> are immutable.
 * An angle can be obtained through the factory methods {@link #fromDegrees} and
 * {@link #fromRadians}.
 * <p>
 * @author Tom Gaskins
 * @author Kyle Johnston (additions to original implementation)
 * @version $Id$
 */
public class Angle implements Serializable {

    private static final long serialVersionUID = -6107478700048091667L;

    /**
     * Represents an angle of zero degrees
     */
    public static final Angle ZERO = Angle.fromDegrees(0);

    /**
     * Represents a right angle of positive 90 degrees
     */
    public static final Angle POS90 = Angle.fromDegrees(90);

    /**
     * Represents a right angle of negative 90 degrees
     */
    public static final Angle NEG90 = Angle.fromDegrees(-90);

    /**
     * Represents an angle of positive 180 degrees
     */
    public static final Angle POS180 = Angle.fromDegrees(180);

    /**
     * Represents an angle of negative 180 degrees
     */
    public static final Angle NEG180 = Angle.fromDegrees(-180);

    /**
     * Represents an angle of positive 360 degrees
     */
    public static final Angle POS360 = Angle.fromDegrees(360);

    /**
     * Represents an angle of negative 360 degrees
     */
    public static final Angle NEG360 = Angle.fromDegrees(-360);

    /**
     * Represents an angle of 1 minute
     */
    public static final Angle MINUTE = Angle.fromDegrees(1d / 60d);

    /**
     * Represents an angle of 1 second
     */
    public static final Angle SECOND = Angle.fromDegrees(1d / 3600d);

    // ===============================
    public static final Angle POSINF
            = Angle.fromDegrees(Double.POSITIVE_INFINITY);

    public static final Angle NEGINF
            = Angle.fromDegrees(Double.NEGATIVE_INFINITY);

    public static final Angle NAN
            = Angle.fromDegrees(Double.NaN);

    public static final double LOG2 = Math.log(2);

    // ====================================
    private static final double DEGREES_TO_RADIANS = Math.PI / 180d;
    private static final double RADIANS_TO_DEGREES = 180d / Math.PI;

    /**
     *
     */
    public final double degrees;

    /**
     *
     */
    public final double radians;

    private static final double ONEHALFPI = Math.PI / 2;

    /**
     *
     * @param angle
     */
    public Angle(Angle angle) {
        this.degrees = angle.degrees;
        this.radians = angle.radians;
    }

    private Angle(double degrees, double radians) {
        this.degrees = degrees;
        this.radians = radians;
    }

    /**
     * Obtains an angle from a specified number of degrees.
     * <p>
     * @param degrees the size in degrees of the angle to be obtained
     * <p>
     * @return a new angle, whose size in degrees is given by
     * <code>degrees</code>
     */
    public static Angle fromDegrees(double degrees) {
        return new Angle(degrees, DEGREES_TO_RADIANS * degrees);
    }

    /**
     * Obtains an angle from a specified number of radians.
     * <p>
     * @param radians the size in radians of the angle to be obtained.
     * <p>
     * @return a new angle, whose size in radians is given by
     * <code>radians</code>.
     */
    public static Angle fromRadians(double radians) {
        return new Angle(RADIANS_TO_DEGREES * radians, radians);
    }

    public static Angle fromDegreesToPositive(double degrees) {

        degrees = degrees % 360.0;
        if (degrees < 0.0) {
            degrees += 360.0;
        }
        return new Angle(degrees, DEGREES_TO_RADIANS * degrees);
    }

    /**
     *
     * @param degrees
     * <p>
     * @return
     */
    public static Angle fromDegreesLatitude(double degrees) {
        double degreesContained = degrees < -90 ? -90 : degrees > 90 ? 90 : degrees;
        double radians = DEGREES_TO_RADIANS * degreesContained;
        radians = radians < -ONEHALFPI ? -ONEHALFPI : radians > ONEHALFPI ? ONEHALFPI : radians;

        return new Angle(degreesContained, radians);
    }

    /**
     *
     * @param radians
     * <p>
     * @return
     */
    public static Angle fromRadiansLatitude(double radians) {
        double radiansContained = radians < -ONEHALFPI ? -ONEHALFPI : radians > ONEHALFPI ? ONEHALFPI : radians;
        double degrees = RADIANS_TO_DEGREES * radiansContained;
        degrees = degrees < -90 ? -90 : degrees > 90 ? 90 : degrees;

        return new Angle(degrees, radiansContained);
    }

    /**
     *
     * @param degrees
     * <p>
     * @return
     */
    public static Angle fromDegreesLongitude(double degrees) {
        double degreesContained = degrees < -180 ? -180 : degrees > 180 ? 180 : degrees;
        double radians = DEGREES_TO_RADIANS * degreesContained;
        radians = radians < -Math.PI ? -Math.PI : radians > Math.PI ? Math.PI : radians;

        return new Angle(degreesContained, radians);
    }

    /**
     *
     * @param radians
     * <p>
     * @return
     */
    public static Angle fromRadiansLongitude(double radians) {
        double radiansContained = radians < -Math.PI ? -Math.PI : radians > Math.PI ? Math.PI : radians;
        double degrees = RADIANS_TO_DEGREES * radiansContained;
        degrees = degrees < -180 ? -180 : degrees > 180 ? 180 : degrees;

        return new Angle(degrees, radiansContained);
    }

    /**
     * Obtains an angle from rectangular coordinates.
     * <p>
     * @param x the abscissa coordinate.
     * @param y the ordinate coordinate.
     * <p>
     * @return a new angle, whose size is determined from <code>x</code> and
     * <code>y</code>.
     */
    public static Angle fromXY(double y, double x) {
        double radians = Math.atan2(y, x);
        return new Angle(RADIANS_TO_DEGREES * radians, radians);
    }

    /**
     * Retrieves the size of this angle in degrees. This method may be faster
     * than first obtaining the radians and then converting to degrees.
     * <p>
     * @return the size of this angle in degrees.
     */
    public final double getDegrees() {
        return this.degrees;
    }

    /**
     * Retrieves the size of this angle in radians. This may be useful for
     * <code>java.lang.Math</code> functions, which generally take radians as
     * trigonometric arguments. This method may be faster that first obtaining
     * the degrees and then converting to radians.
     * <p>
     * @return the size of this angle in radians.
     */
    public final double getRadians() {
        return this.radians;
    }

    /**
     * Multiplies this angle by <code>multiplier</code>. This angle remains
     * unchanged. The result is returned as a new angle.
     * <p>
     * @param multiplier a scalar by which this angle is multiplied.
     * <p>
     * @return a new angle whose size equals this angle's size multiplied by
     * <code>multiplier</code>.
     */
    public final Angle multiply(double multiplier) {
        return Angle.fromDegrees(this.degrees * multiplier);
    }

    /**
     *
     * @param degrees
     * <p>
     * @return
     */
    public final Angle addDegrees(double degrees) {
        return Angle.fromDegrees(this.degrees + degrees);
    }

    /**
     *
     * @param degrees
     * <p>
     * @return
     */
    public final Angle subtractDegrees(double degrees) {
        return Angle.fromDegrees(this.degrees - degrees);
    }

    /**
     * Divides this angle by <code>divisor</code>. This angle remains unchanged.
     * The result is returned as a new angle. Behaviour is undefined if
     * <code>divisor</code> equals zero.
     * <p>
     * @param divisor the number to be divided by.
     * <p>
     * @return a new angle equivalent to this angle divided by
     * <code>divisor</code>.
     */
    public final Angle divide(double divisor) {
        return Angle.fromDegrees(this.degrees / divisor);
    }

    /**
     *
     * @param radians
     * <p>
     * @return
     */
    public final Angle addRadians(double radians) {
        return Angle.fromRadians(this.radians + radians);
    }

    /**
     *
     * @param radians
     * <p>
     * @return
     */
    public final Angle subtractRadians(double radians) {
        return Angle.fromRadians(this.radians - radians);
    }

    public static final Angle subtract(Angle a, Angle b) {
        return a.subtractRadians(b.radians);
    }

    public static final Angle add(Angle a, Angle b) {
        return a.addRadians(b.radians);
    }

    /**
     * Obtains the sine of this angle.
     * <p>
     * @return the trigonometric sine of this angle.
     */
    public final double sin() {
        return Math.sin(this.radians);
    }

    /**
     *
     * @return
     */
    public final double sinHalfAngle() {
        return Math.sin(0.5 * this.radians);
    }

    /**
     *
     * @param sine
     * <p>
     * @return
     */
    public static Angle asin(double sine) {
        return Angle.fromRadians(Math.asin(sine));
    }

    /**
     * Obtains the cosine of this angle.
     * <p>
     * @return the trigonometric cosine of this angle.
     */
    public final double cos() {
        return Math.cos(this.radians);
    }

    /**
     *
     * @return
     */
    public final double cosHalfAngle() {
        return Math.cos(0.5 * this.radians);
    }

    /**
     *
     * @param cosine
     * <p>
     * @return
     */
    public static Angle acos(double cosine) {
        return Angle.fromRadians(Math.acos(cosine));
    }

    /**
     * Obtains the tangent of half of this angle.
     * <p>
     * @return the trigonometric tangent of half of this angle.
     */
    public final double tanHalfAngle() {
        return Math.tan(0.5 * this.radians);
    }

    /**
     *
     * @param tan
     * <p>
     * @return
     */
    public static Angle atan(double tan) {
        return Angle.fromRadians(Math.atan(tan));
    }

    // ==========================================================
    /**
     * Returns the inverse hyperbolic cosine of the specified argument. The
     * inverse hyperbolic cosine is defined as: acosh(x) = log(x + sqrt(
     * (x-1)(x+1) )
     *
     * @param x Value to return inverse hyperbolic cosine of.
     * @return The inverse hyperbolic cosine of x.
     * @throws IllegalArgumentException if x is less than 1.0.
     *
     */
    public static Angle acosh(double x) {
        if (Double.isNaN(x)) {
            return Angle.NAN;
        }
        if (Double.isInfinite(x)) {
            return Angle.POSINF;
        }
        if (x < 1.0) {
            throw new IllegalArgumentException("x may not be less than 1.0");
        }

        double y;
        if (x > 1.0E8) {
            y = Math.log(x) + LOG2;

        } else {
            double a = Math.sqrt((x - 1.0) * (x + 1.0));
            y = Math.log(x + a);
        }

        return Angle.fromRadians(y);
    }

    /**
     * Returns the inverse hyperbolic sine of the specified argument. The
     * inverse hyperbolic sine is defined as: asinh(x) = log( x + sqrt(1 + x*x)
     * )
     *
     * @param xx Value to return inverse hyperbolic cosine of.
     * @return The inverse hyperbolic sine of x.
     *
     */
    public static Angle asinh(double xx) {
        if (Double.isNaN(xx)) {
            return Angle.NAN;
        }
        if (Double.isInfinite(xx)) {
            return Angle.POSINF;
        }
        if (isApproxZero(xx)) {
            return Angle.ZERO;
        }

        int sign = 1;
        double x = xx;
        if (xx < 0) {
            sign = -1;
            x = -xx;
        }

        double y;
        if (x > 1.0E8) {
            y = sign * (Math.log(x) + LOG2);

        } else {
            double a = Math.sqrt(x * x + 1.0);
            y = sign * Math.log(x + a);
        }

        return Angle.fromRadians(y);
    }

    /**
     * Returns the inverse hyperbolic tangent of the specified argument. The
     * inverse hyperbolic tangent is defined as: atanh(x) = 0.5 * log( (1 +
     * x)/(1 - x) )
     *
     * @param x Value to return inverse hyperbolic cosine of.
     * @return The inverse hyperbolic tangent of x.
     * @throws IllegalArgumentException if x is outside the range -1, to +1.
     *
     */
    public static Angle atanh(double x) {
        if (Double.isNaN(x)) {
            return Angle.NAN;
        }
        if (isApproxZero(x)) {
            return Angle.ZERO;
        }

        double z = Math.abs(x);
        if (z >= 1.0) {
            if (isApproxEqual(x, 1.0)) {
                return Angle.POSINF;
            }
            if (isApproxEqual(x, -1.0)) {
                return Angle.NEGINF;
            }

            throw new IllegalArgumentException("x outside of range -1 to +1");
        }

        if (z < 1.0E-7) {
            return Angle.fromRadians(x);
        }

        double y = 0.5 * Math.log((1.0 + x) / (1.0 - x));

        return Angle.fromRadians(y);
    }
    // ============================================================

    /**
     *
     * @param value
     * <p>
     * @return
     */
    public static boolean isValidLatitude(double value) {
        return value >= -90 && value <= 90;
    }

    /**
     *
     * @param value
     * <p>
     * @return
     */
    public static boolean isValidLongitude(double value) {
        return value >= -180 && value <= 180;
    }

    /**
     *
     * @param a
     * @param b
     * <p>
     * @return
     */
    public static Angle max(Angle a, Angle b) {
        return a.degrees >= b.degrees ? a : b;
    }

    /**
     *
     * @param a
     * @param b
     * <p>
     * @return
     */
    public static Angle min(Angle a, Angle b) {
        return a.degrees <= b.degrees ? a : b;
    }

    /**
     * Obtains a <code>String</code> representation of this angle.
     * <p>
     * @return the value of this angle in degrees and as a <code>String</code>.
     */
    @Override
    public final String toString() {
        return Double.toString(this.degrees) + '\u00B0';
    }

    /**
     * Obtains a {@link String} representation of this {@link Angle} formated as
     * degrees, minutes and seconds integer values.
     * <p>
     * @return the value of this angle in degrees, minutes, seconds as a string.
     */
    public final String toDMSString() {
        double temp = this.degrees;
        int sign = (int) Math.signum(temp);
        temp *= sign;
        int d = (int) Math.floor(temp);
        temp = (temp - d) * 60d;
        int m = (int) Math.floor(temp);
        temp = (temp - m) * 60d;
        int s = (int) Math.round(temp);

        if (isApproxEqual(s, 60)) {
            m++;
            s = 0;
        } // Fix rounding errors
        if (isApproxEqual(m, 60)) {
            d++;
            m = 0;
        }

        return (sign == -1 ? "-" : "") + d + '\u00B0' + ' ' + m + '\u2019' + ' ' + s + '\u201d';
    }

    /**
     *
     * @return
     */
    public final String toFormattedDMSString() {
        double[] dms = toDMS();

        return String.format("%4f\u00B0 %2f\u2019 %5.2f\u201d", dms[0], dms[1], dms[2]);
    }

    /**
     *
     * @return
     */
    public final double[] toDMS() {
        double temp = this.degrees;
        int sign = (int) Math.signum(temp);

        temp *= sign;
        int d = (int) Math.floor(temp);
        temp = (temp - d) * 60d;
        int m = (int) Math.floor(temp);
        temp = (temp - m) * 60d;
        double s = Math.rint(temp * 100) / 100;  // keep two decimals for seconds

        if (Math.abs(s - 60.0) <= Math.ulp(1.0)) {
            m++;
            s = 0;
        }

        if (Math.abs(m - 60.0) <= Math.ulp(1.0)) {
            d++;
            m = 0;
        }

        return new double[]{sign * d, m, s};
    }

    /**
     * Obtains the amount of memory this {@link Angle} consumes.
     * <p>
     * @return the memory footprint of this angle in bytes.
     */
    public long getSizeInBytes() {
        return (long) Double.SIZE / 8;
    }

    /**
     *
     * @param o
     * <p>
     * @return
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        Angle angle = (Angle) o;

        //noinspection RedundantIfStatement
        if (isApproxEqual(angle.degrees, this.degrees)) {
            return false;
        }

        return true;
    }

    /**
     * Returns true if the two supplied numbers are approximately equal to
     * within machine precision.
     * <p>
     * @param a
     * @param b
     * <p>
     * @return
     */
    public static boolean isApproxEqual(double a, double b) {
        double eps2 = Math.ulp(a);
        double eps = (eps2 > Math.ulp(1.0) ? eps2 : Math.ulp(1.0));
        return Math.abs(a - b) <= eps;
    }

    /**
     * Returns true if the supplied number is approximately zero to within
     * machine precision.
     * <p>
     * @param a
     * <p>
     * @return
     */
    public static boolean isApproxZero(double a) {
        return Math.abs(a) <= Math.ulp(1.0);
    }

    /**
     *
     * @return
     */
    @Override
    public int hashCode() {
        long temp = isApproxZero(degrees) ? Double.doubleToLongBits(degrees) : 0L;
        return (int) (temp ^ (temp >>> 32));
    }
}
