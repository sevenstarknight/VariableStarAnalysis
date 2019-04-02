/*
 * Copyright (C) 2018 Kyle Johnston
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
package fit.astro.vsa.common.utilities.io.dataset;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import java.util.ArrayList;
import java.util.List;
import java.util.PrimitiveIterator.OfDouble;
import java.util.Random;
import java.util.stream.DoubleStream;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author kjohnston
 */
public class MatlabFunctionsTest {

    public MatlabFunctionsTest() {
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
    public void testWriteOut() {

        Random random = new Random(42L);
        OfDouble streamD = random.doubles().iterator();

        double[][] randomData = new double[10][10];
        for (int idx = 0; idx < 10; idx++) {
            for (int jdx = 0; jdx < 10; jdx++) {
                randomData[idx][jdx] = streamD.nextDouble();
            }
        }

        MLArray errorMLWith = new MLDouble("test", randomData);

        List<MLArray> list = new ArrayList<>();
        list.add(errorMLWith);

        MatlabFunctions.storeToTest("WriteOutTestFunction", list);
    }
}
