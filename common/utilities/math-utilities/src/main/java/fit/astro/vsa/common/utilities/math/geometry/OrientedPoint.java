/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.math.geometry;

import java.util.Comparator;
import fit.astro.vsa.common.utilities.math.NumericTests;

/**
 * The <tt>Point</tt> class is an immutable data type to encapsulate a
 * two-dimensional point with real-value coordinates.
 * <p>
 * Note: in order to deal with the difference behavior of double and Double with
 * respect to -0.0 and +0.0, the Point2D constructor converts any coordinates
 * that are -0.0 to +0.0.
 * <p>
 * For additional documentation, see
 * <a href="http://algs4.cs.princeton.edu/12oop">Section 1.2</a> of
 * <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 * <p>
 * @author Robert Sedgewick
 * @author Kevin Wayne
 */
public class OrientedPoint implements Comparable<OrientedPoint> {

    /**
     * Compares two points by x-coordinate.
     */
    public static final Comparator<OrientedPoint> X_ORDER = new XOrder();

    /**
     * Compares two points by y-coordinate.
     */
    public static final Comparator<OrientedPoint> Y_ORDER = new YOrder();

    /**
     * Compares two points by polar radius.
     */
    public static final Comparator<OrientedPoint> R_ORDER = new ROrder();

    private final double x;    // x coordinate
    private final double y;    // y coordinate

    /**
     * Initializes a new point (x, y).
     * <p>
     * @param x the x-coordinate
     * @param y the y-coordinate
     * <p>
     * @throws IllegalArgumentException if either <tt>x</tt> or <tt>y</tt>
     * is <tt>Double.NaN</tt>, <tt>Double.POSITIVE_INFINITY</tt> or
     * <tt>Double.NEGATIVE_INFINITY</tt>
     */
    public OrientedPoint(double x, double y) {
        if (Double.isInfinite(x) || Double.isInfinite(y)) {
            throw new IllegalArgumentException("Coordinates must be finite");
        }
        if (Double.isNaN(x) || Double.isNaN(y)) {
            throw new IllegalArgumentException("Coordinates cannot be NaN");
        }
        if (NumericTests.isApproxZero(x)) {
            this.x = 0.0;  // convert -0.0 to +0.0
        } else {
            this.x = x;
        }

        if (NumericTests.isApproxZero(y)) {
            this.y = 0.0;  // convert -0.0 to +0.0
        } else {
            this.y = y;
        }
    }

    /**
     * Returns the x-coordinate.
     * <p>
     * @return the x-coordinate
     */
    public double x() {
        return x;
    }

    /**
     * Returns the y-coordinate.
     * <p>
     * @return the y-coordinate
     */
    public double y() {
        return y;
    }

    /**
     * Returns the polar radius of this point.
     * <p>
     * @return the polar radius of this point in polar coordiantes: sqrt(x*x +
     * y*y)
     */
    public double r() {
        return Math.sqrt(x * x + y * y);
    }

    /**
     * Returns the angle of this point in polar coordinates.
     * <p>
     * @return the angle (in radians) of this point in polar coordiantes
     * (between -pi/2 and pi/2)
     */
    public double theta() {
        return Math.atan2(y, x);
    }

    /**
     * Returns the angle between this point and that point.
     * <p>
     * @return the angle in radians (between -pi and pi) between this point and
     * that point (0 if equal)
     */
    private double angleTo(OrientedPoint that) {
        double dx = that.x - this.x;
        double dy = that.y - this.y;
        return Math.atan2(dy, dx);
    }

    /**
     * Returns true if a->b->c is a counterclockwise turn.
     * <p>
     * @param a first point
     * @param b second point
     * @param c third point
     * <p>
     * @return { -1, 0, +1 } if a->b->c is a { clockwise, collinear;
     * counter-clock-wise } turn.
     */
    public static int ccw(OrientedPoint a, OrientedPoint b, OrientedPoint c) {
        double area2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (area2 < 0) {
            return -1;
        } else if (area2 > 0) {
            return +1;
        } else {
            return 0;
        }
    }

    /**
     * Returns twice the signed area of the triangle a-b-c.
     * <p>
     * @param a first point
     * @param b second point
     * @param c third point
     * <p>
     * @return twice the signed area of the triangle a-b-c
     */
    public static double area2(OrientedPoint a, OrientedPoint b, OrientedPoint c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    }

    /**
     * Returns the Euclidean distance between this point and that point.
     * <p>
     * @param that the other point
     * <p>
     * @return the Euclidean distance between this point and that point
     */
    public double distanceTo(OrientedPoint that) {
        double dx = this.x - that.x;
        double dy = this.y - that.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Returns the square of the Euclidean distance between this point and that
     * point.
     * <p>
     * @param that the other point
     * <p>
     * @return the square of the Euclidean distance between this point and that
     * point
     */
    public double distanceSquaredTo(OrientedPoint that) {
        double dx = this.x - that.x;
        double dy = this.y - that.y;
        return dx * dx + dy * dy;
    }

    /**
     * Compares two points by y-coordinate, breaking ties by x-coordinate.Formally, the invoking point (x0, y0) is less than the argument point
 (x1, y1) if and only if either y0 < y1 or if y0 = y1 and x0 < x1.<p>
     * @param that
     * @
     * <p>
     * param that the other point
     * <p>
     * @return the value <tt>0</tt> if this string is equal to the argument
     * string (precisely when <tt>equals()</tt> returns <tt>true</tt>); a
     * negative integer if this point is less than the argument point; and a
     * positive integer if this point is greater than the argument point
     */
    @Override
    public int compareTo(OrientedPoint that) {
        if (this.y < that.y) {
            return -1;
        }
        if (this.y > that.y) {
            return +1;
        }
        if (this.x < that.x) {
            return -1;
        }
        if (this.x > that.x) {
            return +1;
        }
        return 0;
    }

    /**
     * Compares two points by polar angle (between 0 and 2pi) with respect to
     * this point.
     * <p>
     * @return the comparator
     */
    public Comparator<OrientedPoint> polarOrder() {
        return new PolarOrder();
    }

    /**
     * Compares two points by atan2() angle (between -pi and pi) with respect to
     * this point.
     * <p>
     * @return the comparator
     */
    public Comparator<OrientedPoint> atan2Order() {
        return new Atan2Order();
    }

    /**
     * Compares two points by distance to this point.
     * <p>
     * @return the comparator
     */
    public Comparator<OrientedPoint> distanceToOrder() {
        return new DistanceToOrder();
    }

    // compare points according to their x-coordinate
    private static class XOrder implements Comparator<OrientedPoint> {

        @Override
        public int compare(OrientedPoint p, OrientedPoint q) {
            if (p.x < q.x) {
                return -1;
            }
            if (p.x > q.x) {
                return +1;
            }
            return 0;
        }
    }

    // compare points according to their y-coordinate
    private static class YOrder implements Comparator<OrientedPoint> {

        @Override
        public int compare(OrientedPoint p, OrientedPoint q) {
            if (p.y < q.y) {
                return -1;
            }
            if (p.y > q.y) {
                return +1;
            }
            return 0;
        }
    }

    // compare points according to their polar radius
    private static class ROrder implements Comparator<OrientedPoint> {

        @Override
        public int compare(OrientedPoint p, OrientedPoint q) {
            double delta = (p.x * p.x + p.y * p.y) - (q.x * q.x + q.y * q.y);
            if (delta < 0) {
                return -1;
            }
            if (delta > 0) {
                return +1;
            }
            return 0;
        }
    }

    // compare other points relative to atan2 angle (bewteen -pi/2 and pi/2) they make with this Point
    private class Atan2Order implements Comparator<OrientedPoint> {

        @Override
        public int compare(OrientedPoint q1, OrientedPoint q2) {
            double angle1 = angleTo(q1);
            double angle2 = angleTo(q2);
            if (angle1 < angle2) {
                return -1;
            } else if (angle1 > angle2) {
                return +1;
            } else {
                return 0;
            }
        }
    }

    // compare other points relative to polar angle (between 0 and 2pi) they make with this Point
    private class PolarOrder implements Comparator<OrientedPoint> {

        @Override
        public int compare(OrientedPoint q1, OrientedPoint q2) {
            double dx1 = q1.x - x;
            double dy1 = q1.y - y;
            double dx2 = q2.x - x;
            double dy2 = q2.y - y;

            if (dy1 >= 0 && dy2 < 0) {
                return -1;    // q1 above; q2 below
            } else if (dy2 >= 0 && dy1 < 0) {
                return +1;    // q1 below; q2 above
            } else if (NumericTests.isApproxZero(dy1)
                    && NumericTests.isApproxZero(dy2)) {            // 3-collinear and horizontal
                if (dx1 >= 0 && dx2 < 0) {
                    return -1;
                } else if (dx2 >= 0 && dx1 < 0) {
                    return +1;
                } else {
                    return 0;
                }
            } else {
                return -ccw(OrientedPoint.this, q1, q2);     // both above or below
            }
            // Note: ccw() recomputes dx1, dy1, dx2, and dy2
        }
    }

    // compare points according to their distance to this point
    private class DistanceToOrder implements Comparator<OrientedPoint> {

        @Override
        public int compare(OrientedPoint p, OrientedPoint q) {
            double dist1 = distanceSquaredTo(p);
            double dist2 = distanceSquaredTo(q);
            if (dist1 < dist2) {
                return -1;
            } else if (dist1 > dist2) {
                return +1;
            } else {
                return 0;
            }
        }
    }

    /**
     * Compares this point to the specified point.
     * <p>
     * @param other the other point
     * <p>
     * @return <tt>true</tt> if this point equals <tt>other</tt>;
     * <tt>false</tt> otherwise
     */
    @Override
    public boolean equals(Object other) {
        if (other == this) {
            return true;
        }
        if (other == null) {
            return false;
        }
        if (other.getClass() != this.getClass()) {
            return false;
        }
        OrientedPoint that = (OrientedPoint) other;
        return NumericTests.isApproxEqual(this.x, that.x)
                && NumericTests.isApproxEqual(this.y, that.y);
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 71 * hash + (int) (Double.doubleToLongBits(this.x) ^ (Double.doubleToLongBits(this.x) >>> 32));
        hash = 71 * hash + (int) (Double.doubleToLongBits(this.y) ^ (Double.doubleToLongBits(this.y) >>> 32));
        return hash;
    }

}
