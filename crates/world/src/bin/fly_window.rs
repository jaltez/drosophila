//! Windowed viewer: brain-driven fly in an odor arena, rendered live.
//!
//! Real-time pacing: 16 sim-ms per frame (~60 fps). The Sim core is
//! engine-independent; this binary only observes and draws it.

use std::sync::Arc;


use bevy::prelude::*;
use brain::LifParams;
use connectome::Connectome;
use transduce::{BehaviorMapFile, OdorField};
use world::{Sim, WorldConfig};

const PIXELS_PER_MM: f32 = 10.0;
const TICKS_PER_FRAME: u64 = 16;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("bridge/out/connectome-malecns-v1.0");
    let params_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("params/homeostasis_v1.json");
    let map_path = args
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("scenarios/behavior_map.json");
    let odor_x: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(25.0);
    let odor_y: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let seed: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(7);

    let con = Connectome::load(std::path::Path::new(con_dir)).expect("load connectome");
    let mut params: LifParams =
        serde_json::from_slice(&std::fs::read(params_path).unwrap()).unwrap();
    params.seed = seed;
    let map: BehaviorMapFile =
        serde_json::from_slice(&std::fs::read(map_path).unwrap()).unwrap();
    let steer_insts: Vec<transduce::SteerInstance> = serde_json::from_slice(
        &std::fs::read("scenarios/steer_instances.json").unwrap(),
    )
    .unwrap();
    let cfg = WorldConfig {
        seed,
        ..WorldConfig::for_con(&con)
    };
    let field = OdorField {
        x: odor_x,
        y: odor_y,
        strength: 1.0,
        radius_mm: 8.0,
    };
    let mut sim = Sim::new(Arc::new(con), params, &map, field, cfg);
    sim.apply_steer_instances(&steer_insts);

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "drosophila — MaleCNS embodied".into(),
                resolution: bevy::window::WindowResolution::new(900, 900),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(SimBox(Some(sim)))
        .insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(Update, (advance, draw))
        .run();
}

#[derive(Resource)]
struct SimBox(Option<Sim>);

#[derive(Resource)]
struct Entities {
    fly: Entity,
    arena: Entity,
    odor: Entity,
    text: Entity,
}

fn setup(mut commands: Commands, sim: Res<SimBox>, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<ColorMaterial>>) {
    commands.spawn(Camera2d);
    let sim = sim.0.as_ref().unwrap();

    let r_arena = sim_cfg(sim).arena_radius_mm as f32 * PIXELS_PER_MM;
    let arena = commands
        .spawn((
            Mesh2d(meshes.add(Circle::new(r_arena))),
            MeshMaterial2d(materials.add(Color::srgb(0.09, 0.12, 0.16))),
        ))
        .id();

    let r_odor = 4.0 * PIXELS_PER_MM;
    let odor = commands
        .spawn((
            Mesh2d(meshes.add(Circle::new(r_odor))),
            MeshMaterial2d(materials.add(Color::srgb(0.1, 0.5, 0.2))),
            Transform::from_xyz(
                sim.field.x as f32 * PIXELS_PER_MM,
                sim.field.y as f32 * PIXELS_PER_MM,
                0.0,
            ),
        ))
        .id();

    let fly = commands
        .spawn((
            Mesh2d(meshes.add(Circle::new(1.2 * PIXELS_PER_MM))),
            MeshMaterial2d(materials.add(Color::srgb(0.95, 0.8, 0.2))),
        ))
        .id();

    let text = commands
        .spawn((
            Text::new("booting"),
            TextFont {
                font_size: FontSize::Px(22.0),
                ..default()
            },
            TextColor(Color::srgb(0.9, 0.9, 0.9)),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                ..default()
            },
        ))
        .id();

    commands.insert_resource(Entities {
        fly,
        arena,
        odor,
        text,
    });
}

fn sim_cfg(sim: &Sim) -> &WorldConfig {
    sim.config()
}

fn advance(mut sim: ResMut<SimBox>) {
    let s = sim.0.as_mut().unwrap();
    for _ in 0..TICKS_PER_FRAME {
        s.step();
    }
}

fn draw(
    sim: ResMut<SimBox>,
    ents: Option<Res<Entities>>,
    mut transforms: Query<&mut Transform>,
    mut texts: Query<&mut Text>,
) {
    let Some(ents) = ents else { return };
    let s = sim.0.as_ref().unwrap();
    if let Ok(mut t) = transforms.get_mut(ents.fly) {
        t.translation.x = s.fly.x as f32 * PIXELS_PER_MM;
        t.translation.y = s.fly.y as f32 * PIXELS_PER_MM;
        t.rotation = Quat::from_rotation_z(s.fly.heading as f32);
    }
    if let Ok(mut t) = transforms.get_mut(ents.odor) {
        t.translation.x = s.field.x as f32 * PIXELS_PER_MM;
        t.translation.y = s.field.y as f32 * PIXELS_PER_MM;
    }
    if let Ok(mut text) = texts.get_mut(ents.text) {
        let d = s.drives();
        let (dx, dy) = (s.fly.x - s.field.x, s.fly.y - s.field.y);
        **text = format!(
            "walk {:+.1} Hz  steer {:+.1} Hz  esc {:.1} Hz\npos ({:.0},{:.0}) mm  dist to odor {:.0} mm",
            d.walk,
            d.steer,
            d.escape,
            s.fly.x,
            s.fly.y,
            (dx * dx + dy * dy).sqrt()
        );
    }
}
