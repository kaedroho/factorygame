//! A simple 3D scene with light shining over a cube sitting on a plane.

use std::f32::consts::PI;

use bevy::{prelude::*, render::{render_resource::{PrimitiveTopology, Face}, mesh::Indices}};
use ndarray::Array2;
use noise::{Perlin, NoiseFn};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup_environment)
        .add_startup_system(setup_terrain)
        .add_startup_system(setup_camera)
        .add_system(update_camera)
        .run();
}

const SKY_COLOR: Color = Color::Hsla { hue: 213.0, saturation: 0.3, lightness: 0.95, alpha: 1.0 };
const SUN_COLOR: Color = Color::rgb(1.0, 1.0, 1.0);

fn setup_environment(mut commands: Commands) {
    // Sun
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight { color: SUN_COLOR, illuminance: 30000.0, ..default() },
        transform: Transform { rotation: Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -0.2), ..default() },
        ..default()
    });

    // Sky
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight { color: SKY_COLOR, illuminance: 10000.0, ..default() },
        transform: Transform { rotation: Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -1.5), ..default() },
        ..default()
    });
}

pub enum GetHeightMethod {
    Nearest,
    Bilinear,
}

const CHUNK_DETAIL: u32 = 16;
const STEP_HEIGHT: f32 = 1.0;
const CHUNK_SIZE: f32 = 512.0;

struct Terrain {
    pub heightmap: Array2<u16>,
}

pub struct ChunkMesh {
    pub vertex_positions: Vec<(f32, f32, f32)>,
    pub vertex_normals: Vec<(f32, f32, f32)>,
    pub indices: Vec<u32>,
    pub min_positions: [f32; 3],
    pub max_positions: [f32; 3],
}

impl Terrain {
    pub fn get_height(&self, x: i32, y: i32) -> Option<f32> {
        if x > 0 && x < self.heightmap.shape()[0] as i32 && y > 0 && y < self.heightmap.shape()[1] as i32 {
            Some(self.heightmap[[x as usize, y as usize]] as f32 * STEP_HEIGHT)
        } else {
            None
        }
    }

    pub fn get_normal(&self, x: i32, y: i32) -> Option<(f32, f32, f32)> {
        let x0 = self.get_height(x - 1, y);
        let x1 = self.get_height(x + 1, y);
        let y0 = self.get_height(x, y - 1);
        let y1 = self.get_height(x, y + 1);

        if let (Some(x0), Some(x1), Some(y0), Some(y1)) = (x0, x1, y0, y1) {
            // Create a normal vector
            let nx = x0 - x1;
            let ny = y0 - y1;
            let nz = 2.0;

            // Normalize the vector
            let mag = (nx * nx + ny * ny + nz * nz).sqrt();
            Some((nx / mag, ny / mag, nz / mag))
        } else {
            // At least one height is missing
            None
        }
    }

    pub fn generate_chunk_mesh(
        &self,
        chunk_x: u32,
        chunk_y: u32,
        chunk_world_x: f32,
        chunk_world_y: f32,
        scale: f32,
    ) -> ChunkMesh {
        // Generate vertex data
        let num_vertices = (CHUNK_DETAIL + 1) * (CHUNK_DETAIL + 1);
        let num_indices = CHUNK_DETAIL * CHUNK_DETAIL * 6;
        let mut vertex_positions: Vec<(f32, f32, f32)> = Vec::with_capacity(num_vertices as usize);
        let mut vertex_normals: Vec<(f32, f32, f32)> = Vec::with_capacity(num_vertices as usize);
        let mut indices: Vec<u32> = Vec::with_capacity(num_indices as usize);

        // Add vertices
        for vertex_y in 0..(CHUNK_DETAIL + 1) {
            for vertex_x in 0..(CHUNK_DETAIL + 1) {
                let px = (chunk_x * CHUNK_DETAIL + vertex_x) as i32;
                let py = (chunk_y * CHUNK_DETAIL + vertex_y) as i32;

                // Get height
                let z = self
                    .get_height(
                        px,
                        py,
                    )
                    .unwrap_or_default();

                // Scale X/Y
                let x = vertex_x as f32 * scale + chunk_world_x;
                let y = vertex_y as f32 * scale + chunk_world_y;

                // Get normals
                let (nx, ny, nz) = self.get_normal(px, py).unwrap_or((0.0, 0.0, 1.0));

                vertex_positions.push((x, z, y));
                vertex_normals.push((nx, ny, nz));
            }
        }

        // Get position mins and maxes
        let mut min_positions = [f32::MAX, f32::MAX, f32::MAX];
        let mut max_positions = [f32::MIN, f32::MIN, f32::MIN];
        for (x, y, z) in &vertex_positions {
            if *x < min_positions[0] {
                min_positions[0] = *x;
            }

            if *y < min_positions[1] {
                min_positions[1] = *y;
            }

            if *z < min_positions[2] {
                min_positions[2] = *z;
            }

            if *x > max_positions[0] {
                max_positions[0] = *x;
            }

            if *y > max_positions[1] {
                max_positions[1] = *y;
            }

            if *z > max_positions[2] {
                max_positions[2] = *z;
            }
        }

        // Add indices
        for tile_y in 0..CHUNK_DETAIL {
            for tile_x in 0..CHUNK_DETAIL {
                let tl_i = tile_y * (CHUNK_DETAIL + 1) + tile_x;
                let tr_i = tile_y * (CHUNK_DETAIL + 1) + (tile_x + 1);
                let bl_i = (tile_y + 1) * (CHUNK_DETAIL + 1) + tile_x;
                let br_i = (tile_y + 1) * (CHUNK_DETAIL + 1) + (tile_x + 1);

                let tl_height = vertex_positions[tl_i as usize].2;
                let tr_height = vertex_positions[tr_i as usize].2;
                let bl_height = vertex_positions[bl_i as usize].2;
                let br_height = vertex_positions[br_i as usize].2;

                // Rotate the tiles to reduce artifacts on cliff-edges that go diagonally across the terrain
                if (tl_height - br_height).abs() > (tr_height - bl_height).abs() {
                    indices.extend([tl_i, tr_i, bl_i, tr_i, br_i, bl_i].into_iter());
                } else {
                    indices.extend([tl_i, tr_i, br_i, tl_i, br_i, bl_i].into_iter());
                }
            }
        }

        ChunkMesh {
            vertex_positions,
            vertex_normals,
            indices,
            min_positions,
            max_positions,
        }
    }
}

// Terrain size in chunks
const TERRAIN_SIZE: (u32, u32) = (16, 16);

const TILE_SIZE: f32 = CHUNK_SIZE / CHUNK_DETAIL as f32;

pub fn setup_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {

    let noise_generator = Perlin::new(1);
    let terrain = Terrain {
        heightmap: Array2::from_shape_fn(((TERRAIN_SIZE.0 * CHUNK_DETAIL) as usize, (TERRAIN_SIZE.1 * CHUNK_DETAIL) as usize), |(x, y)| {
            ((noise_generator.get([x as f64 * TILE_SIZE as f64 * 0.001, y as f64 * TILE_SIZE as f64 * 0.001, 20.8]) * 100.0) / STEP_HEIGHT as f64).floor() as u16
        }),
    };

    let mut chunks = Vec::with_capacity((TERRAIN_SIZE.0 * TERRAIN_SIZE.1) as usize);

    // Middle bit
    for y_chunk in 0..TERRAIN_SIZE.0 {
        for x_chunk in 0..TERRAIN_SIZE.1 {
            let mesh = terrain.generate_chunk_mesh(
                x_chunk,
                y_chunk,
                x_chunk as f32 * CHUNK_SIZE,
                y_chunk as f32 * CHUNK_SIZE,
                CHUNK_SIZE / CHUNK_DETAIL as f32,
            );

            chunks.push(mesh);
        }
    }

    for chunk in chunks {
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            chunk.vertex_positions.iter().map(|(x, y, z)| Vec3::new(*x, *y, *z)).collect::<Vec<Vec3>>(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            chunk.vertex_normals.iter().map(|(x, y, z)| Vec3::new(*x, *y, *z)).collect::<Vec<Vec3>>(),
        );
        mesh.set_indices(Some(Indices::U32(chunk.indices)));

        commands.spawn((PbrBundle {
            mesh: meshes.add(mesh),
            material: materials.add(StandardMaterial {
                base_color: Color::rgb(0.3, 0.5, 0.3).into(),
                cull_mode: Some(Face::Front),
                ..default()
            }),
            ..default()
        },));
    }
}


#[derive(Component, Debug)]
pub struct CameraState {
    position: Vec2,
    velocity: Vec2,
    target_position: Vec2,
    zoom: f32,
    zoom_velocity: f32,
    target_zoom: f32,
}

impl Default for CameraState {
    fn default() -> CameraState {
        CameraState {
            position: Vec2::new(1000.0, 1000.0),
            velocity: Vec2::new(0.0, 0.0),
            target_position: Vec2::new(1000.0, 1000.0),
            zoom: 1.0,
            zoom_velocity: 0.0,
            target_zoom: 1.0,
        }
    }
}

impl CameraState {
    pub fn move_worldspace(&mut self, vector: Vec2) {
        let angle = vector.y.atan2(vector.x);
        let distance = vector.length();
        let sin = angle.sin();
        let cos = angle.cos();

        self.target_position.x += distance * sin * self.zoom;
        self.target_position.y += distance * cos * self.zoom;
    }

    pub fn move_screenspace(&mut self, mut vector: Vec2) {
        let speed = vector.length();
        if speed < 0.005 {
            return;
        }

        vector /= speed;
        vector.x /= 16.0/9.0;
        vector.y = -vector.y;
        vector *= (PI / 8.0).tan();

        let raystart = Vec3::new(vector.x * 10.0, vector.y * 10.0, 10.0);
        let rayend = Vec3::new(vector.x * 10000.0, vector.y * 10000.0, 10000.0);

        let m = self.get_view_matrix().inverse();
        let raystart = m.transform_point3(raystart);
        let rayend =  m.transform_point3(rayend);

        let raydiff = rayend - raystart;
        let ray_xz_slope = raydiff.x / raydiff.z;
        let ray_yz_slope = raydiff.y / raydiff.z;

        let x = raystart.x - ray_xz_slope * raystart.z;
        let y = raystart.y - ray_yz_slope * raystart.z;

        let intersection_point = Vec2::new(x, y);
        let move_vector = -(intersection_point - self.position).normalize() * speed * (self.zoom.powf(1.5));

        self.target_position += move_vector;

    }

    pub fn zoom(&mut self, amount: f32) {
        self.target_zoom += amount;

        if self.target_zoom < 0.5 {
            self.target_zoom = 0.5
        }

        if self.target_zoom > 3.0 {
            self.target_zoom = 3.0
        }
    }

    pub fn set_zoom(&mut self, mut zoom: f32) {
        if zoom < 0.5 {
            zoom = 0.5;
        }

        if zoom > 3.0 {
            zoom = 3.0;
        }

        self.target_zoom = zoom;
        self.zoom = zoom;
    }

    pub fn get_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(PI / 4.0, 16.0/9.0, 10.0, 10000.0)
    }

    pub fn get_world_position(&self) -> Vec3 {
        let center = Vec3::new(self.position.y, 0.0, self.position.x);
        let offset = Vec3::new(-1000.0, 1000.0, 0.0) * self.zoom * self.zoom;
        center + offset
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        let center = Vec3::new(self.position.x, self.position.y, 0.0);
        let eye = self.get_world_position();

        Mat4::look_at_rh(eye, center, Vec3::new(0.0, 0.0, 1.0))
    }

    pub fn screen_to_world(&self, mut position: Vec3) -> Vec3 {
        position.x *= 16.0 / 9.0;
        position.y = -position.y;
        position.x *= (PI / 8.0).tan();
        position.y *= (PI / 8.0).tan();

        let position = Vec3::new(position.x * position.z, position.y * position.z, position.z);

        self.get_view_matrix().inverse().transform_point3(position)
    }

    pub fn update(&mut self) {
        let target_velocity = self.target_position - self.position;
        let velocity_difference = target_velocity - self.velocity;
        self.velocity += velocity_difference * 0.25;
        self.position += self.velocity * 0.1;

        let target_zoom_velocity = self.target_zoom - self.zoom;
        let zoom_velocity_difference = target_zoom_velocity - self.zoom_velocity;
        self.zoom_velocity += zoom_velocity_difference * 0.25;
        self.zoom = self.zoom + self.zoom_velocity * 0.1;
    }
}

fn setup_camera(
    mut commands: Commands
) {
    commands.spawn((Camera3dBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    }, CameraState::default()));
}

fn update_camera(
    mut query: Query<(&mut Transform, &mut CameraState)>,
    keys: Res<Input<KeyCode>>,
) {
    let mut camera_state = query.single_mut().1;

    // Keyboard camera movement
    let mut camera_move_vector = Vec2::ZERO;
    for key in keys.get_pressed() {
            match key {
            KeyCode::W => camera_move_vector.y -= 1.0,
            KeyCode::S => camera_move_vector.y += 1.0,
            KeyCode::A => camera_move_vector.x -= 1.0,
            KeyCode::D => camera_move_vector.x += 1.0,
            _ => (),
        }
    }
    let magnitude = camera_move_vector.length();
    if magnitude > 0.0 {
        // Normalise the vector
        camera_move_vector /= magnitude;

        // Set speed
        camera_move_vector *= 10.0;

        camera_state.move_worldspace(camera_move_vector);
    }

    camera_state.update();

    for (mut transform, camera_state) in query.iter_mut() {
        transform.translation = camera_state.get_world_position();
        transform.rotation = Quat::from_rotation_x(-PI / 4.0);
    }
}
