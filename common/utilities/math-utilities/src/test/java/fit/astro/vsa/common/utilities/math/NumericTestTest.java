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
package fit.astro.vsa.common.utilities.math;

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
public class NumericTestTest {

    public NumericTestTest() {
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
    public void testNumericTestsEvenOdd() {

        /**
         * Test the set of even/odd algorithm logic
         */
        assertEquals(Boolean.TRUE, NumericTests.isEven(100L));
        assertEquals(Boolean.FALSE, NumericTests.isOdd(100L));

    }

    @Test
    public void testNumericTestsZero() {
        /**
         * Test the is equal logic
         */
        assertEquals(Boolean.FALSE, NumericTests.isApproxEqual(0, 2));
        assertEquals(Boolean.TRUE, NumericTests.isApproxEqual(10e-19, 10e-18));

        assertEquals(Boolean.TRUE, NumericTests.isApproxZero(0));
        assertEquals(Boolean.TRUE, NumericTests.isApproxZero(10e-32));
        assertEquals(Boolean.FALSE, NumericTests.isApproxZero(10e-16));
    }

    @Test
    public void testNumericTestsSign() {

        assertEquals(-10, NumericTests.sign(10, -10), 0.0);

    }
}
