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

/**
 * Represents vertices of the hull, as well as the points from which it is
 * formed.
 *
 * @author John E. Lloyd, Fall 2004
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu> (Edits and Updates)
 */
public class Vertex extends Point3D {

    /**
     * Back index into an array.
     */
    public int index;

    /**
     * List forward link.
     */
    public Vertex prev;

    /**
     * List backward link.
     */
    public Vertex next;

    /**
     * Current face that this vertex is outside of.
     */
    public HullFace face;

    /**
     * Constructs a vertex and sets its coordinates to 0.
     */
    public Vertex() {
        super(new Point3D());
    }

    /**
     * Constructs a vertex with the specified coordinates and index.
     *
     * @param x
     * @param y
     * @param z
     * @param idx
     */
    public Vertex(double x, double y, double z, int idx) {
        super(new Point3D(x, y, z));
        index = idx;
    }

    /**
     *
     * @param v
     * @param idx
     */
    public Vertex(double[] v, int idx) {
        super(new Point3D(v));
        index = idx;
    }

    /**
     *
     * @param v
     * @param idx
     */
    public Vertex(Point3D v, int idx) {
        super(new Point3D(v));
        index = idx;
    }
}
