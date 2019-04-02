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
package fit.astro.vsa.common.utilities.math;

import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
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
public class SortingFunctionTest {

    public SortingFunctionTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    @Test
    public void testSortingMaps() {

        Random random = new Random(42L);

        double center = random.nextDouble();
        Map<Integer, Double> unsortedDistance = new HashMap<>();

        for (int idx = 0; idx < 100; idx++) {
            unsortedDistance.put(idx,
                    Math.abs(center - random.nextDouble()));
        }

        // ==============================
        Map<Integer, Double> sortedAcendingDistances
                = SortingOperations
                .sortByAcendingValue(unsortedDistance);

        double start = 0.0;
        for (Integer jdx : sortedAcendingDistances.keySet()) {

            double distance = sortedAcendingDistances.get(jdx);
            boolean doesWork;
            if (start < distance) {
                doesWork = Boolean.TRUE;
                start = distance;
            } else {
                doesWork = Boolean.FALSE;
            }

            assertEquals(Boolean.TRUE, doesWork);
        }

        // ==============================
        Map<Integer, Double> sortedDecendingDistances
                = SortingOperations
                .sortByDecendingValue(unsortedDistance);

        start = 10000.0;
        for (Integer jdx : sortedDecendingDistances.keySet()) {

            double distance = sortedDecendingDistances.get(jdx);
            boolean doesWork;
            if (start > distance) {
                doesWork = Boolean.TRUE;
                start = distance;
            } else {
                doesWork = Boolean.FALSE;
            }

            assertEquals(Boolean.TRUE, doesWork);
        }

    }
}
