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
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class OrientedGrahamScanTest {

    private List<OrientedPoint> pts;

    public OrientedGrahamScanTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {

        Random rnd = new Random(42L);

        this.pts = new ArrayList<>();
        for (int idx = 0; idx < 100; idx++) {
            RealVector xy = RandomSurfaceGenerator.getPointOnUnitCircle(rnd);
            pts.add(new OrientedPoint(xy.getEntry(0), xy.getEntry(1)));

        }

        // make a random circle + origin
        pts.add(new OrientedPoint(0.0, 0.0));
    }

    @After
    public void tearDown() {
    }

    @Test
    public void testOrientedGrahamScan() {

        OrientedGrahamScan grahamScan = new OrientedGrahamScan(pts);
        Iterable<OrientedPoint> hull = grahamScan.hull();

        List<OrientedPoint> target = new ArrayList<>();
        hull.forEach(target::add);

        //hull should be one less than the pts (removal of origin)
        assertEquals(pts.size() - 1, target.size(), 0);

        //order of points should be in ccw radial
        Iterator<OrientedPoint> iterPts = hull.iterator();
        OrientedPoint aPt = iterPts.next();
        OrientedPoint bPt = iterPts.next();

        while (iterPts.hasNext()) {
            OrientedPoint cPt = iterPts.next();

            assertEquals(OrientedPoint.ccw(aPt, bPt, cPt), +1);

            aPt = bPt;
            bPt = cPt;
        }

    }
}
