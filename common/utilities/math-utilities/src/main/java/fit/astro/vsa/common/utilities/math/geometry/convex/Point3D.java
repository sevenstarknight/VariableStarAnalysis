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
package fit.astro.vsa.common.utilities.math.geometry.convex;

import java.util.Random;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

/**
 * A three-element spatial point.
 *
 * The only difference between a point and a vector is in the the way it is
 * transformed by an affine transformation.
 *
 * @author John E. Lloyd, Fall 2004
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu> (Edits and Updates)
 */
public class Point3D extends Vector3D {

    /**
     * Creates a Point3d and initializes it to zero.
     */
    public Point3D() {
        super(0.0, 0.0, 0.0);
    }

    /**
     *
     * @param v
     */
    public Point3D(double[] v) {
        super(v);
    }

    /**
     * Creates a Point3d by copying a vector
     *
     * @param v vector to be copied
     */
    public Point3D(Vector3D v) {
        super(v.toArray());
    }

    /**
     * Creates a Point3d with the supplied element values.
     *
     * @param x first element
     * @param y second element
     * @param z third element
     */
    public Point3D(double x, double y, double z) {
        super(x, y, z);
    }

    /**
     * Sets the elements of this vector to uniformly distributed random values
     * in a specified range, using a supplied random number generator.
     *
     * @param lower lower random value (inclusive)
     * @param upper upper random value (exclusive)
     * @param generator random number generator
     * @return
     */
    public static Point3D setRandom(double lower, double upper,
            Random generator) {
        double range = upper - lower;

        double x = generator.nextDouble() * range + lower;
        double y = generator.nextDouble() * range + lower;
        double z = generator.nextDouble() * range + lower;

        return new Point3D(x, y, z);
    }
}
