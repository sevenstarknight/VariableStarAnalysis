/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.math.geometry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Stack;

/**
 * Graham, R. L. (1972). An efficient algorithm for determining the convex hull
 * of a finite planar set. Information processing letters, 1(4), 132-133.
 * http://algs4.cs.princeton.edu/99hull/GrahamScan.java
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class OrientedGrahamScan {

    // =============================================
    // Output
    private Stack<OrientedPoint> hull = new Stack<>();

    /**
     * Computes the convex hull of the specified array of points.
     * <p>
     * @param pts the array of points
     * <p>
     * @throws NullPointerException if <tt>points</tt> is <tt>null</tt> or if
     * any entry in <tt>points[]</tt> is <tt>null</tt>
     */
    public OrientedGrahamScan(List<OrientedPoint> pts) {

        // defensive copy
        List<OrientedPoint> points = new ArrayList<>(pts);

        // preprocess so that points[0] has lowest y-coordinate; break ties by x-coordinate
        // points[0] is an extreme point of the convex hull
        // (alternatively, could do easily in linear time)
        Collections.sort(points);

        // sort by polar angle with respect to base point points[0],
        // breaking ties by distance to points[0]
        Collections.sort(points, points.get(0).polarOrder());

        hull.push(points.get(0));       // p[0] is first extreme point

        // find index k1 of first point not equal to points[0]
        int k1 = 0;
        for (OrientedPoint op : points) {
            if (!points.get(0).equals(op)) {
                k1 = points.indexOf(op);
                break;
            }
        }

        // find index k2 of first point not collinear with points[0] and points[k1]
        int k2 = 0;
        for (OrientedPoint op : points) {
            if (!points.get(0).equals(op) && !points.get(k1).equals(op)) {
                if (OrientedPoint.ccw(points.get(0), points.get(k1), op) != 0) {
                    k2 = points.indexOf(op) - 1;
                    break;
                }
            }
        }

        hull.push(points.get(k2));    // points[k2-1] is second extreme point

        // Graham scan; note that points[N-1] is extreme point different from points[0]
        for (OrientedPoint op : points) {
            if (!points.get(0).equals(op) && !points.get(k1).equals(op)) {
                OrientedPoint top = hull.pop();
                while (OrientedPoint.ccw(hull.peek(), top, op) <= 0) {
                    top = hull.pop();
                }
                hull.push(top);
                hull.push(op);
            }
        }

        assert isConvex();
    }

    /**
     * Returns the extreme points on the convex hull in counterclockwise order.
     * <p>
     * @return the extreme points on the convex hull in counterclockwise order
     */
    public Iterable<OrientedPoint> hull() {
        Stack<OrientedPoint> s = new Stack<>();
        for (OrientedPoint p : hull) {
            s.push(p);
        }
        return s;
    }

    /**
     * check that boundary of hull is strictly convex
     */
    private boolean isConvex() {
        int N = hull.size();
        if (N <= 2) {
            return true;
        }

        OrientedPoint[] points = new OrientedPoint[N];
        int n = 0;
        for (OrientedPoint p : hull()) {
            points[n++] = p;
        }

        for (int i = 0; i < N; i++) {
            if (OrientedPoint.ccw(points[i], points[(i + 1) % N], points[(i + 2) % N]) <= 0) {
                return false;
            }
        }
        return true;
    }

}
