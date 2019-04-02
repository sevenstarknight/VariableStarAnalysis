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

import fit.astro.vsa.common.utilities.math.geometry.convex.shapes.Hexahedron;
import java.util.Set;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Test the Qhull algorithm with defined polyhedra vertices
 * @author Kyle B. Johnston
 */
public class QuickHullTest {

    /**
     * 
     */
    public QuickHullTest() {
    }

    /**
     * Test the method against a d4
     */
    @Test
    public void testQuickHull() {

        // x y z coordinates of 6 points
        Point3D[] points = new Point3D[]{new Point3D(0.0, 0.0, 0.0),
            new Point3D(1.0, 0.5, 0.0),
            new Point3D(2.0, 0.0, 0.0),
            new Point3D(0.5, 0.5, 0.5),
            new Point3D(0.0, 0.0, 2.0),
            new Point3D(0.1, 0.2, 0.3),
            new Point3D(0.0, 2.0, 0.0),};

        QuickHull3D hull = new QuickHull3D();
        hull.build(points);

        Point3D[] vertices = hull.getVertices();

        assertEquals(vertices.length, 4);

        int[][] faceIndices = hull.getHullFaces();

        assertEquals(faceIndices.length, 4);

    }

    /**
     * Test the method against a Hexahedron
     */
    @Test
    public void testQuickHull_Hex() {

        Polyhedron poly1 = new Hexahedron(new Vector3D(-2, -2, -2),
                new Vector3D(2, 2, 2), 0.5);

        Set<Point3D> setPoints = poly1.getVertices();
        Point3D[] array = setPoints.toArray(new Point3D[setPoints.size()]);

        QuickHull3D hull = new QuickHull3D();
        hull.build(array);

        Point3D[] vertices = hull.getVertices();

        assertEquals(vertices.length, 8);

        int[][] faceIndices = hull.getHullFaces();

        int numFaces = hull.getNumHullFaces();
        assertEquals(faceIndices.length, 8);

    }

    /**
     * Test the method against a Prymid
     */
    @Test
    public void testQuickHull_Pry() {

        // x y z coordinates of 6 points
        Point3D[] points = new Point3D[]{new Point3D(0.0, 0.0, 0.0),
            new Point3D(0.0, 0.0, 1.0),
            new Point3D(1.0, 0.0, 1.0),
            new Point3D(1.0, 0.0, 0.0),
            new Point3D(1.0, 1.0, 1.0)};

        QuickHull3D hull = new QuickHull3D();
        hull.build(points);

        Point3D[] vertices = hull.getVertices();

        assertEquals(vertices.length, 5);

        int[][] faceIndices = hull.getHullFaces();
        

        assertEquals(faceIndices.length, 5);

    }

}
