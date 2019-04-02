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

import fit.astro.vsa.common.bindings.math.vector.ModFunction;
import fit.astro.vsa.common.bindings.math.vector.PowerFunction;
import fit.astro.vsa.common.bindings.math.vector.FixFunction;
import fit.astro.vsa.common.bindings.math.vector.RoundFunction;
import fit.astro.vsa.common.bindings.math.vector.Base10Function;
import fit.astro.vsa.common.bindings.math.vector.MinFunction;
import fit.astro.vsa.common.bindings.math.vector.MaxFunction;
import org.apache.commons.math3.linear.MatrixUtils;
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
public class ArrayMapTesting {

    public ArrayMapTesting() {
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

    private static final RealVector IN_VECTOR_A
            = MatrixUtils.createRealVector(new double[]{4.0, 2.0, 3.0});

    @Test
    public void testArrayMappingOperations() {

        RealVector mod = IN_VECTOR_A.map(new ModFunction(2));
        assertArrayEquals(new double[]{0., 0.0, 1.0}, mod.toArray(), 0.001);
    }

    @Test
    public void testArrayMappingPower() {

        RealVector base10 = IN_VECTOR_A.map(new Base10Function());
        assertArrayEquals(new double[]{1e4, 1e2, 1e3}, base10.toArray(), 0.001);

        RealVector power = IN_VECTOR_A.map(new PowerFunction(2.0));
        assertArrayEquals(new double[]{16.0, 4.0, 9.0}, power.toArray(), 0.001);

    }

    @Test
    public void testArrayMappingManip() {

        RealVector fix = IN_VECTOR_A.map(new FixFunction());
        assertArrayEquals(new double[]{4.0, 2.0, 3.0}, fix.toArray(), 0.001);

        RealVector round = IN_VECTOR_A.map(new RoundFunction());
        assertArrayEquals(new double[]{4.0, 2.0, 3.0}, round.toArray(), 0.001);

    }

    @Test
    public void testArrayMappingConditional() {
        RealVector max = IN_VECTOR_A.map(new MaxFunction(3.2));
        assertArrayEquals(new double[]{4.0, 3.2, 3.2}, max.toArray(), 0.001);

        RealVector min = IN_VECTOR_A.map(new MinFunction(3.2));
        assertArrayEquals(new double[]{3.2, 2.0, 3.0}, min.toArray(), 0.001);

    }
}
