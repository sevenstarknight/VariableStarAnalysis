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

import fit.astro.vsa.common.utilities.math.geometry.convex.shapes.Pyramid;
import fit.astro.vsa.common.utilities.math.geometry.convex.shapes.Tetrahedron;
import fit.astro.vsa.common.utilities.math.geometry.convex.shapes.Hexahedron;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

import org.junit.Test;
import static org.junit.Assert.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston
 */
public class SutherlandHodgman3DTest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(SutherlandHodgman3DTest.class);

    public SutherlandHodgman3DTest() {
    }

    @Test
    public  void StoreTetra() {
        Tetrahedron poly1 = new Tetrahedron(new Vector3D(-2, -2, -2),
                new Vector3D(2, 2, 2));
        Tetrahedron poly2 = new Tetrahedron(new Vector3D(-1, -5, -1),
                new Vector3D(1, 5, 1));

        Polyhedron update = poly2.clip(poly1);

        assertEquals(11, update.getVertices().size());
    }

    @Test
    public  void StorePyramid() {
        Pyramid poly1 = new Pyramid(new Vector3D(-2, -2, -2),
                new Vector3D(2, 2, 2));
        Pyramid poly2 = new Pyramid(new Vector3D(-1, -5, -1),
                new Vector3D(1, 5, 1));

        Polyhedron update = poly1.clip(poly2);

        assertEquals(12, update.getVertices().size());
    }

    @Test
    public  void StoreHexahedron() {
        Polyhedron poly1 = new Hexahedron(new Vector3D(-2, -2, -2),
                new Vector3D(0, 0, 0), 0.5);
        Polyhedron poly2 = new Hexahedron(new Vector3D(-1, -5, -1),
                new Vector3D(1, 5, 1), -2.5);

        Polyhedron update = poly1.clip(poly2);

        assertEquals(15, update.getVertices().size());
    }

}
